"""
Treina dois classificadores especializados em XGBoost + CUDA usando a saída do
modelo publicado do texto2area.

Pipeline:
    texto cru
        -> texto2area (9 grandes áreas)
        -> se CIÊNCIAS HUMANAS: XGBoost especializado (9 áreas de avaliação)
        -> se LINGUÍSTICA, LETRAS E ARTES: XGBoost especializado (2 áreas)
        -> caso contrário: nenhuma cabeça especializada é acionada.

Para treino, o corpus `dados/corpus_td_lemas.parquet` é usado. A coluna
`lemmas_ext` já está no formato pré-processado consumido pelo vetorizador do
texto2area, então o script não refaz normalização/lemmatização/n-gramas.

Features do segundo estágio, derivadas da saída do texto2area:
    1. 9 margens de `decision_function`;
    2. grande área prevista, em one-hot (9 atributos);
    3. até N termos decisivos, em hashing esparso, com prefixo de ranking.

Balanceamento:
    - undersampling aleatório exato antes do split;
    - todas as classes de cada especialista ficam com a quantidade da menor
      classe da própria base (ex.: 805 em Humanas e 1924 em LLA, conforme
      os dados observados);
    - o split posterior é estratificado.

Instalação:
    pip install -U xgboost tqdm pandas pyarrow scipy scikit-learn joblib

Exemplos:
    python reproduzir/treinar_xgboost_humanas_lla.py --amostra 10000
    python reproduzir/treinar_xgboost_humanas_lla.py --amostra 100000 --n-estimators 300
    python reproduzir/treinar_xgboost_humanas_lla.py

Depois do treino, testar texto cru:
    python reproduzir/treinar_xgboost_humanas_lla.py --sem-treinamento \
        --texto "Título. Resumo do trabalho acadêmico..."
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    import xgboost as xgb
except ImportError as exc:
    raise SystemExit("Instale XGBoost com: pip install -U xgboost") from exc


# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CORPUS = REPO / "dados" / "corpus_td_lemas.parquet"
T2A_DATA = REPO / "texto2area" / "data"
T2A_VECTORIZER = T2A_DATA / "vetorizador.joblib"
T2A_MODEL = T2A_DATA / "modelo.joblib"
OUT_DIR = HERE / "modelo_treinado" / "especialistas_xgboost"

SEED = 42


# ---------------------------------------------------------------------------
# Taxonomia do experimento
# ---------------------------------------------------------------------------

HUMANAS = "CIÊNCIAS HUMANAS"
LLA = "LINGUÍSTICA, LETRAS E ARTES"

CLASSES_HUMANAS = [
    "ANTROPOLOGIA / ARQUEOLOGIA",
    "CIÊNCIA POLÍTICA E RELAÇÕES INTERNACIONAIS",
    "CIÊNCIAS DA RELIGIÃO E TEOLOGIA",
    "EDUCAÇÃO",
    "FILOSOFIA",
    "GEOGRAFIA",
    "HISTÓRIA",
    "PSICOLOGIA",
    "SOCIOLOGIA",
]

CLASSES_LLA = [
    "ARTES",
    "LINGUÍSTICA E LITERATURA",
]

# Configuração padrão pensando em uma GPU com 6 GB.
DEFAULT_BATCH = 25_000
DEFAULT_TOP_TERMS = 5
DEFAULT_HASH_FEATURES = 2_048
DEFAULT_N_ESTIMATORS = 500
DEFAULT_MAX_DEPTH = 6
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_MIN_CHILD_WEIGHT = 5
DEFAULT_SUBSAMPLE = 0.80
DEFAULT_COLSAMPLE = 0.80
DEFAULT_MAX_BIN = 256
DEFAULT_EARLY_STOPPING = 50
DEFAULT_TEST_SIZE = 0.15
DEFAULT_VAL_SIZE = 0.15


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def normalizar_rotulo(valor: object) -> str:
    """Normaliza espaços/Unicode e leva o rótulo para caixa alta."""
    if pd.isna(valor):
        return ""
    texto = str(valor).replace("\u00a0", " ")
    texto = unicodedata.normalize("NFKC", texto)
    return " ".join(texto.strip().split()).upper()


def mapear_area_especialista(valor: object) -> str | None:
    """
    Converte os rótulos históricos encontrados no corpus nas classes usadas
    neste experimento.

    Humanas: os nove nomes correntes são mantidos. Formas históricas de
    Teologia/Filosofia são incorporadas à classe corrente correspondente.

    LLA: ARTES / MÚSICA -> ARTES e LETRAS / LINGUÍSTICA ->
    LINGUÍSTICA E LITERATURA.
    """
    area = normalizar_rotulo(valor)

    mapa = {
        # LLA
        "ARTES": "ARTES",
        "ARTES / MÚSICA": "ARTES",
        "LINGUÍSTICA E LITERATURA": "LINGUÍSTICA E LITERATURA",
        "LINGUISTICA E LITERATURA": "LINGUÍSTICA E LITERATURA",
        "LETRAS / LINGUÍSTICA": "LINGUÍSTICA E LITERATURA",
        "LETRAS / LINGUISTICA": "LINGUÍSTICA E LITERATURA",

        # Humanas
        "ANTROPOLOGIA / ARQUEOLOGIA": "ANTROPOLOGIA / ARQUEOLOGIA",
        "CIÊNCIA POLÍTICA E RELAÇÕES INTERNACIONAIS":
            "CIÊNCIA POLÍTICA E RELAÇÕES INTERNACIONAIS",
        "CIENCIA POLITICA E RELACOES INTERNACIONAIS":
            "CIÊNCIA POLÍTICA E RELAÇÕES INTERNACIONAIS",
        "CIÊNCIAS DA RELIGIÃO E TEOLOGIA":
            "CIÊNCIAS DA RELIGIÃO E TEOLOGIA",
        "CIENCIAS DA RELIGIAO E TEOLOGIA":
            "CIÊNCIAS DA RELIGIÃO E TEOLOGIA",
        "TEOLOGIA": "CIÊNCIAS DA RELIGIÃO E TEOLOGIA",
        "FILOSOFIA/TEOLOGIA:SUBCOMISSÃO TEOLOGIA":
            "CIÊNCIAS DA RELIGIÃO E TEOLOGIA",
        "EDUCAÇÃO": "EDUCAÇÃO",
        "EDUCACAO": "EDUCAÇÃO",
        "FILOSOFIA": "FILOSOFIA",
        "FILOSOFIA/TEOLOGIA:SUBCOMISSÃO FILOSOFIA": "FILOSOFIA",
        "GEOGRAFIA": "GEOGRAFIA",
        "HISTÓRIA": "HISTÓRIA",
        "HISTORIA": "HISTÓRIA",
        "PSICOLOGIA": "PSICOLOGIA",
        "SOCIOLOGIA": "SOCIOLOGIA",
    }
    return mapa.get(area)


def titulo(texto: str) -> None:
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78, flush=True)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def carregar_corpus(batch_size: int, amostra: int | None) -> pd.DataFrame:
    """Lê só as colunas necessárias e filtra Humanas + LLA em streaming."""
    if not CORPUS.exists():
        raise FileNotFoundError(
            f"Corpus não encontrado: {CORPUS}\n"
            "Baixe o `corpus_td_lemas.parquet` do release do repositório."
        )

    import pyarrow.parquet as pq

    pf = pq.ParquetFile(CORPUS)
    total = pf.metadata.num_rows
    colunas = ["id_producao", "grande_area", "area_avaliacao", "lemmas_ext"]

    partes: list[pd.DataFrame] = []
    mantidos = 0

    with tqdm(
        total=total,
        desc="Lendo corpus",
        unit=" docs",
        dynamic_ncols=True,
    ) as bar:
        for tabela in pf.iter_batches(batch_size=batch_size, columns=colunas):
            bloco = tabela.to_pandas()

            grande = bloco["grande_area"].map(normalizar_rotulo)
            bloco = bloco.loc[grande.isin({HUMANAS, LLA})].copy()
            if bloco.empty:
                bar.update(len(tabela))
                continue

            bloco["grande_area"] = grande.loc[bloco.index]
            bloco["area_classe"] = bloco["area_avaliacao"].map(mapear_area_especialista)

            bloco = bloco.dropna(subset=["lemmas_ext", "area_classe"])
            bloco = bloco[bloco["lemmas_ext"].str.strip() != ""]

            mask_h = (
                (bloco["grande_area"] == HUMANAS)
                & bloco["area_classe"].isin(CLASSES_HUMANAS)
            )
            mask_lla = (
                (bloco["grande_area"] == LLA)
                & bloco["area_classe"].isin(CLASSES_LLA)
            )
            bloco = bloco.loc[mask_h | mask_lla]

            if not bloco.empty:
                partes.append(bloco)
                mantidos += len(bloco)

            bar.update(len(tabela))
            if amostra is not None and mantidos >= amostra:
                break

    if not partes:
        raise RuntimeError("Nenhum documento das duas grandes áreas foi encontrado.")

    df = pd.concat(partes, ignore_index=True)
    if amostra is not None and len(df) > amostra:
        df = df.sample(n=amostra, random_state=SEED).reset_index(drop=True)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Texto2area: reproduz as três saídas publicadas
# ---------------------------------------------------------------------------

def carregar_texto2area():
    """Carrega exatamente os artefatos publicados do texto2area."""
    if not T2A_VECTORIZER.exists():
        raise FileNotFoundError(f"Não encontrei: {T2A_VECTORIZER}")
    if not T2A_MODEL.exists():
        raise FileNotFoundError(f"Não encontrei: {T2A_MODEL}")

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    vec = joblib.load(T2A_VECTORIZER)
    clf = joblib.load(T2A_MODEL)
    return vec, clf


def extrair_termos_decisivos(
    X: sparse.csr_matrix,
    clf,
    nomes_features: np.ndarray,
    pred_idx: np.ndarray,
    topo: int,
) -> list[list[str]]:
    """Reproduz a lógica dos termos decisivos do `_classify.py` do repositório."""
    resultados: list[list[str]] = []
    coef = clf.coef_

    for i in range(X.shape[0]):
        ini = X.indptr[i]
        fim = X.indptr[i + 1]
        if ini == fim:
            resultados.append([])
            continue

        indices = X.indices[ini:fim]
        valores = X.data[ini:fim]
        k = int(pred_idx[i])
        contribuicao = valores * coef[k, indices]
        n = min(topo, contribuicao.size)

        if contribuicao.size > n:
            cand = np.argpartition(contribuicao, -n)[-n:]
        else:
            cand = np.arange(contribuicao.size)

        cand = cand[np.argsort(contribuicao[cand])[::-1]]
        resultados.append([str(nomes_features[indices[j]]) for j in cand])

    return resultados


def features_saida_texto2area(
    textos: pd.Series,
    vec,
    clf,
    batch_size: int,
    top_terms: int,
    hash_features: int,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """
    Constrói as features finais e retorna também a grande área prevista.

    Ordem das features:
        0..8             -> 9 margens, na ordem de `clf.classes_`
        9..17            -> one-hot da grande área prevista
        restante         -> termos decisivos (hashing)
    """
    nomes_features = np.asarray(vec.get_feature_names_out())
    classes = np.asarray(clf.classes_, dtype=object)
    chunks: list[sparse.csr_matrix] = []
    predicoes: list[np.ndarray] = []

    hasher = FeatureHasher(
        n_features=hash_features,
        input_type="string",
        alternate_sign=False,
        dtype=np.float32,
    )

    with tqdm(
        total=len(textos),
        desc="Gerando saída do texto2area",
        unit=" docs",
        dynamic_ncols=True,
    ) as bar:
        for ini in range(0, len(textos), batch_size):
            fim = min(ini + batch_size, len(textos))
            bloco = textos.iloc[ini:fim].to_numpy()

            # `lemmas_ext` já está no formato esperado pelo analyzer publicado.
            X_base = vec.transform(bloco).tocsr()
            margens = np.asarray(clf.decision_function(X_base))
            if margens.ndim == 1:
                margens = margens.reshape(-1, 1)

            idx_pred = np.argmax(margens, axis=1).astype(np.int32)
            pred = classes[idx_pred]
            predicoes.append(pred)

            termos = extrair_termos_decisivos(
                X_base,
                clf,
                nomes_features,
                idx_pred,
                top_terms,
            )

            onehot = sparse.csr_matrix(
                (
                    np.ones(len(idx_pred), dtype=np.float32),
                    (np.arange(len(idx_pred)), idx_pred),
                ),
                shape=(len(idx_pred), len(classes)),
            )

            # O rank faz parte da feature: r1:termo != r2:termo.
            entradas_hash = [
                [f"r{rank}:{term}" for rank, term in enumerate(ts, 1)]
                for ts in termos
            ]
            hashed = hasher.transform(entradas_hash).tocsr()
            X_margin = sparse.csr_matrix(margens.astype(np.float32, copy=False))

            chunks.append(
                sparse.hstack([X_margin, onehot, hashed], format="csr", dtype=np.float32)
            )
            bar.update(len(bloco))

    return sparse.vstack(chunks, format="csr", dtype=np.float32), np.concatenate(predicoes)


# ---------------------------------------------------------------------------
# Callback de progresso do XGBoost
# ---------------------------------------------------------------------------

class TQDMCallback(xgb.callback.TrainingCallback):
    def __init__(self, total: int, desc: str):
        self.total = total
        self.desc = desc
        self.bar = None

    def before_training(self, model):
        self.bar = tqdm(
            total=self.total,
            desc=self.desc,
            unit=" árvores",
            dynamic_ncols=True,
        )
        return model

    def after_iteration(self, model, epoch: int, evals_log):
        if self.bar is None:
            return False

        self.bar.n = min(epoch + 1, self.total)
        try:
            if evals_log:
                dataset = list(evals_log.keys())[-1]
                metricas = evals_log[dataset]
                nome = list(metricas.keys())[-1]
                valor = metricas[nome][-1]
                self.bar.set_postfix(**{nome: f"{valor:.5f}"})
        except Exception:
            pass
        self.bar.refresh()
        return False

    def after_training(self, model):
        if self.bar is not None:
            self.bar.close()
        return model


# ---------------------------------------------------------------------------
# Treinamento/eval de uma cabeça
# ---------------------------------------------------------------------------

def split_estratificado(
    y: np.ndarray,
    test_size: float,
    val_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retorna índices de treino, validação e teste."""
    idx = np.arange(len(y))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx,
        y,
        test_size=test_size + val_size,
        random_state=SEED,
        stratify=y,
    )

    frac_test = test_size / (test_size + val_size)
    idx_val, idx_test, _, _ = train_test_split(
        idx_temp,
        y_temp,
        test_size=frac_test,
        random_state=SEED,
        stratify=y_temp,
    )
    return idx_train, idx_val, idx_test


def balancear_dataset_por_undersampling(
    y: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Faz undersampling aleatório e determinístico antes do split.

    A menor classe de cada especialista define a quantidade-alvo. Portanto,
    depois desta etapa todas as classes do conjunto completo têm exatamente
    o mesmo número de documentos. O split estratificado seguinte mantém essa
    proporção nos conjuntos de treino, validação e teste.

    Exemplo com os números observados pelo usuário:
        Humanas -> alvo = 805 documentos por classe
        LLA     -> alvo = 1802 documentos por classe
    """
    rng = np.random.default_rng(SEED)
    contagens = np.array(
        [np.sum(y == classe) for classe in range(n_classes)],
        dtype=int,
    )
    alvo = int(contagens.min())

    if alvo <= 0:
        raise ValueError(
            "Pelo menos uma classe não possui documentos e não pode ser "
            "balanceada para o mesmo tamanho."
        )

    partes = []
    registros = []

    for classe in range(n_classes):
        idx_classe = np.flatnonzero(y == classe)
        quantidade_antes = len(idx_classe)

        escolhidos = rng.choice(
            idx_classe,
            size=alvo,
            replace=False,
        )
        partes.append(escolhidos)

        registros.append({
            "classe_id": classe,
            "quantidade_antes": quantidade_antes,
            "quantidade_depois": alvo,
            "quantidade_descartada": quantidade_antes - alvo,
            "alvo_balanceamento": alvo,
        })

    idx_balanceado = np.concatenate(partes)
    rng.shuffle(idx_balanceado)

    # Verificação obrigatória: depois do undersampling, todas as classes
    # precisam ter exatamente a mesma quantidade de documentos.
    contagens_depois = np.bincount(
        y[idx_balanceado],
        minlength=n_classes,
    )
    if not np.all(contagens_depois == alvo):
        raise RuntimeError(
            "Falha no balanceamento: as classes não ficaram com a mesma "
            f"quantidade. Esperado={alvo}; obtido={contagens_depois.tolist()}"
        )

    diagnostico = pd.DataFrame(registros)
    diagnostico["quantidade_depois"] = contagens_depois
    return idx_balanceado, diagnostico

def treinar_especialista(
    nome: str,
    X: sparse.csr_matrix,
    labels: np.ndarray,
    classes: list[str],
    out_dir: Path,
    args: argparse.Namespace,
) -> dict:
    titulo(f"XGBOOST — {nome.upper()}")

    classes_to_int = {c: i for i, c in enumerate(classes)}
    inesperadas = sorted(set(labels) - set(classes))
    if inesperadas:
        raise ValueError(f"Classes inesperadas em {nome}: {inesperadas}")

    y_original = np.asarray(
        [classes_to_int[str(v)] for v in labels],
        dtype=np.int32,
    )

    print(f"Documentos originais: {len(y_original):,}")
    print(f"Features:             {X.shape[1]:,}")
    print(f"Classes:              {len(classes)}")
    print("Distribuição original:")
    contagem_original = pd.Series(labels).value_counts().reindex(
        classes, fill_value=0
    )
    for c, n in contagem_original.items():
        print(f"  {c}: {n:,}")

    # O balanceamento é realizado ANTES do split. Assim, garante que
    # todas as classes tenham o tamanho da menor classe, isso vale
    # para a base do especialista como um todo e, por consequência, para
    # treino/validação/teste após o split estratificado.
    if args.balancear_treino:
        idx_balanceado, balanceamento = balancear_dataset_por_undersampling(
            y_original,
            len(classes),
        )
        X = X[idx_balanceado]
        y = y_original[idx_balanceado]
    else:
        balanceamento = pd.DataFrame({
            "classe_id": np.arange(len(classes)),
            "quantidade_antes": [
                int(np.sum(y_original == i)) for i in range(len(classes))
            ],
            "quantidade_depois": [
                int(np.sum(y_original == i)) for i in range(len(classes))
            ],
            "quantidade_descartada": [0] * len(classes),
            "alvo_balanceamento": [np.nan] * len(classes),
        })
        y = y_original

    print("\nDistribuição após balanceamento:")
    contagem_balanceada = pd.Series(y).value_counts().reindex(
        np.arange(len(classes)), fill_value=0
    )
    for i, c in enumerate(classes):
        print(f"  {c}: {int(contagem_balanceada.iloc[i]):,}")

    if args.balancear_treino:
        alvo = int(balanceamento["alvo_balanceamento"].iloc[0])
        print(f"\nUndersampling ativado.")
        print(f"Quantidade-alvo por classe: {alvo:,} documentos")
        print(f"Total após balanceamento:   {len(y):,} documentos")
    else:
        print("\nBalanceamento por undersampling: DESATIVADO")

    idx_train, idx_val, idx_test = split_estratificado(
        y, DEFAULT_TEST_SIZE, DEFAULT_VAL_SIZE
    )

    X_train = X[idx_train]
    X_val = X[idx_val]
    X_test = X[idx_test]
    y_train = y[idx_train]
    y_val = y[idx_val]
    y_test = y[idx_test]

    print(
        f"\nTreino: {len(idx_train):,} | "
        f"Validação: {len(idx_val):,} | "
        f"Teste: {len(idx_test):,}"
    )

    # Humanas tem 9 classes (multiclass) e LLA tem 2 classes (binário).
    # A métrica de validação precisa acompanhar o objetivo:
    #   - binary:logistic -> logloss
    #   - multi:softprob  -> mlogloss
    # Usar logloss no modelo multiclass faz o XGBoost comparar 9 previsões
    # por amostra com apenas 1 rótulo, causando:
    #   preds.Size() == labels.Size()
    # Portanto, a métrica é definida condicionalmente aqui.
    objetivo_binario = len(classes) == 2
    objective = "binary:logistic" if objetivo_binario else "multi:softprob"
    eval_metric = "logloss" if objetivo_binario else "mlogloss"

    params = dict(
        objective=objective,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=0.0,
        reg_lambda=1.0,
        gamma=0.0,
        max_bin=args.max_bin,
        tree_method="hist",
        device="cuda",
        eval_metric=eval_metric,
        random_state=SEED,
        early_stopping_rounds=args.early_stopping,
        callbacks=[TQDMCallback(args.n_estimators, f"Treinando {nome}")],
    )
    if len(classes) > 2:
        params["num_class"] = len(classes)

    print(
        f"\nTreino: {len(idx_train):,} | "
        f"Validação: {len(idx_val):,} | "
        f"Teste: {len(idx_test):,}"
    )
    print("Dispositivo XGBoost: CUDA")

    model = xgb.XGBClassifier(**params)
    t0 = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    tempo = time.time() - t0

    print("\nAvaliando...")
    with tqdm(total=1, desc=f"Avaliando {nome}", unit=" etapa", dynamic_ncols=True) as bar:
        y_pred = model.predict(X_test).astype(np.int32)
        probs = model.predict_proba(X_test)
        bar.update(1)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\nMétricas:")
    print(f"  Accuracy:          {acc:.6f}")
    print(f"  Balanced accuracy: {bal_acc:.6f}")
    print(f"  F1 macro:          {f1_macro:.6f}")
    print(f"  F1 weighted:       {f1_weighted:.6f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=np.arange(len(classes)),
            target_names=classes,
            digits=4,
            zero_division=0,
        )
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    balanceamento_out = balanceamento.copy()
    balanceamento_out["classe"] = [
        classes[int(i)] for i in balanceamento_out["classe_id"]
    ]
    balanceamento_out = balanceamento_out[
        [
            "classe_id",
            "classe",
            "quantidade_antes",
            "quantidade_depois",
            "quantidade_descartada",
            "alvo_balanceamento",
        ]
    ]
    balanceamento_out.to_csv(
        out_dir / f"balanceamento_{nome}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    model_path = out_dir / f"modelo_{nome}.json"
    model.save_model(model_path)

    pred_df = pd.DataFrame({
        "y_true": [classes[int(v)] for v in y_test],
        "y_pred": [classes[int(v)] for v in y_pred],
        "confianca": probs.max(axis=1),
    })
    pred_df.to_csv(
        out_dir / f"predicoes_teste_{nome}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(classes)))
    cm_norm = confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(len(classes)),
        normalize="true",
    )
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(
        out_dir / f"matriz_confusao_{nome}.csv",
        encoding="utf-8-sig",
    )
    pd.DataFrame(cm_norm, index=classes, columns=classes).to_csv(
        out_dir / f"matriz_confusao_normalizada_{nome}.csv",
        encoding="utf-8-sig",
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(classes)),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(
        out_dir / f"classification_report_{nome}.csv",
        encoding="utf-8-sig",
    )

    # Importância por feature do XGBoost, útil para a análise do trabalho.
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    imp_rows = [
        {"feature": k, "gain": float(v)} for k, v in gain.items()
    ]
    if imp_rows:
        pd.DataFrame(imp_rows).sort_values("gain", ascending=False).to_csv(
            out_dir / f"importancia_gain_{nome}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    meta = {
        "nome": nome,
        "classes": classes,
        "n_documentos": int(len(y_original)),
        "n_documentos_original": int(len(y_original)),
        "n_documentos_balanceados": int(len(y)),
        "n_treino": int(len(idx_train)),
        "n_validacao": int(len(idx_val)),
        "n_teste": int(len(idx_test)),
        "balanceamento": {
            "metodo": "undersampling_aleatorio",
            "aplicado_antes_do_split": bool(args.balancear_treino),
            "quantidade_alvo_por_classe": (
                int(balanceamento["alvo_balanceamento"].iloc[0])
                if args.balancear_treino else None
            ),
        },
        "n_features": int(X.shape[1]),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "best_iteration": int(getattr(model, "best_iteration", -1)),
        "best_score": float(getattr(model, "best_score", np.nan)),
        "segundos_treino": round(tempo, 3),
        "params": {
            "objective": objective,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "min_child_weight": args.min_child_weight,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "max_bin": args.max_bin,
            "tree_method": "hist",
            "device": "cuda",
            "early_stopping_rounds": args.early_stopping,
        },
    }
    (out_dir / f"treino_{nome}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {"model": model, "classes": classes, "metadata": meta}


# ---------------------------------------------------------------------------
# Inferência: texto cru -> texto2area -> especialista
# ---------------------------------------------------------------------------

def features_de_saida_unica(
    area: str,
    margens: list[tuple[str, float]],
    termos: list[str],
    classes_grandes: list[str],
    hash_features: int,
) -> sparse.csr_matrix:
    margem_dict = {str(c): float(v) for c, v in margens}
    margens_vetor = np.asarray(
        [margem_dict.get(c, 0.0) for c in classes_grandes],
        dtype=np.float32,
    ).reshape(1, -1)

    if area not in classes_grandes:
        raise ValueError(f"Grande área inesperada: {area}")
    idx = classes_grandes.index(area)

    onehot = sparse.csr_matrix(
        (np.array([1.0], dtype=np.float32), ([0], [idx])),
        shape=(1, len(classes_grandes)),
    )

    hasher = FeatureHasher(
        n_features=hash_features,
        input_type="string",
        alternate_sign=False,
        dtype=np.float32,
    )
    tokens = [[f"r{i}:{t}" for i, t in enumerate(termos, start=1)]]
    hashed = hasher.transform(tokens).tocsr()

    return sparse.hstack(
        [sparse.csr_matrix(margens_vetor), onehot, hashed],
        format="csr",
        dtype=np.float32,
    )


def testar_texto(
    texto: str,
    modelos: dict[str, dict],
    classes_grandes: list[str],
    hash_features: int,
) -> None:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from texto2area import classificar

    area, margens, termos = classificar(texto, topo=DEFAULT_TOP_TERMS)

    titulo("PREDIÇÃO EM TEXTO CRU")
    print(f"Grande área prevista: {area}")
    print("Margens:")
    for c, v in margens:
        print(f"  {c}: {v:.6f}")
    print(f"Termos decisivos: {termos}")

    if area == HUMANAS:
        chave = "ciencias_humanas"
    elif area == LLA:
        chave = "linguistica_letras_artes"
    else:
        print("\nNenhuma das duas cabeças especializadas foi acionada.")
        return

    info = modelos[chave]
    X = features_de_saida_unica(
        area,
        margens,
        termos,
        classes_grandes,
        hash_features,
    )
    pred = int(info["model"].predict(X)[0])
    prob = info["model"].predict_proba(X)[0]

    print(f"\nEspecialista acionado: {chave}")
    print(f"Área de avaliação prevista: {info['classes'][pred]}")
    print("Probabilidades:")
    for c, p in sorted(zip(info["classes"], prob), key=lambda x: -x[1]):
        print(f"  {c}: {p:.6f}")


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--amostra", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--top-termos", type=int, default=DEFAULT_TOP_TERMS)
    ap.add_argument("--hash-features", type=int, default=DEFAULT_HASH_FEATURES)
    ap.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    ap.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    ap.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    ap.add_argument("--min-child-weight", type=int, default=DEFAULT_MIN_CHILD_WEIGHT)
    ap.add_argument("--subsample", type=float, default=DEFAULT_SUBSAMPLE)
    ap.add_argument("--colsample-bytree", type=float, default=DEFAULT_COLSAMPLE)
    ap.add_argument("--max-bin", type=int, default=DEFAULT_MAX_BIN)
    ap.add_argument("--early-stopping", type=int, default=DEFAULT_EARLY_STOPPING)
    ap.add_argument(
        "--sem-balanceamento",
        dest="balancear_treino",
        action="store_false",
        help="desativa o undersampling do treino (para comparação experimental)",
    )
    ap.set_defaults(balancear_treino=True)
    ap.add_argument("--texto", type=str, default=None)
    ap.add_argument(
        "--sem-treinamento",
        action="store_true",
        help="carrega os modelos já salvos e usa somente --texto",
    )
    return ap.parse_args()


def carregar_modelos_salvos() -> dict[str, dict]:
    modelos: dict[str, dict] = {}
    configuracoes = {
        "ciencias_humanas": CLASSES_HUMANAS,
        "linguistica_letras_artes": CLASSES_LLA,
    }
    for nome, classes in configuracoes.items():
        path = OUT_DIR / f"modelo_{nome}.json"
        if not path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {path}")
        model = xgb.XGBClassifier()
        model.load_model(path)
        modelos[nome] = {"model": model, "classes": classes}
    return modelos


def main() -> None:
    args = parse_args()
    inicio_total = time.time()

    if args.batch_size <= 0:
        raise ValueError("--batch-size deve ser > 0")
    if args.top_termos <= 0:
        raise ValueError("--top-termos deve ser > 0")
    if args.hash_features <= 0:
        raise ValueError("--hash-features deve ser > 0")
    if not (0 < args.subsample <= 1):
        raise ValueError("--subsample deve estar em (0, 1]")
    if not (0 < args.colsample_bytree <= 1):
        raise ValueError("--colsample-bytree deve estar em (0, 1]")

    titulo("XGBOOST ESPECIALIZADO — CIÊNCIAS HUMANAS + LLA")
    print(f"Repositório: {REPO}")
    print(f"Corpus:      {CORPUS}")
    print(f"Saída:       {OUT_DIR}")
    print("GPU:         device='cuda'")
    print("Árvores:     tree_method='hist'")
    print(f"Seed:        {SEED}")

    # Apenas inferência.
    if args.sem_treinamento:
        if not args.texto:
            raise SystemExit("--sem-treinamento exige --texto")
        modelos = carregar_modelos_salvos()
        _vec, t2a_clf = carregar_texto2area()
        classes_grandes = [str(c) for c in t2a_clf.classes_]
        testar_texto(
            args.texto,
            modelos,
            classes_grandes,
            args.hash_features,
        )
        return

    # 1) Corpus
    titulo("1/5 — CARREGAMENTO DO CORPUS")
    df = carregar_corpus(args.batch_size, args.amostra)
    print(f"Documentos selecionados: {len(df):,}")
    print("\nGrandes áreas:")
    print(df["grande_area"].value_counts().to_string())
    print("\nÁreas de avaliação:")
    print(df["area_classe"].value_counts().to_string())

    # 2) Saída do texto2area
    titulo("2/5 — GERAÇÃO DA SAÍDA DO TEXTO2AREA")
    vec, clf = carregar_texto2area()
    classes_grandes = [str(c) for c in clf.classes_]
    print(f"Classes do texto2area: {len(classes_grandes)}")
    print(f"Features internas do texto2area: {len(vec.get_feature_names_out()):,}")

    X_all, pred_grande = features_saida_texto2area(
        df["lemmas_ext"],
        vec,
        clf,
        args.batch_size,
        args.top_termos,
        args.hash_features,
    )
    df["grande_area_predita_texto2area"] = pred_grande

    print(f"\nMatriz final: {X_all.shape[0]:,} x {X_all.shape[1]:,}")
    print(f"Elementos não nulos: {X_all.nnz:,}")
    print(
        "Densidade: "
        f"{100 * X_all.nnz / (X_all.shape[0] * X_all.shape[1]):.6f}%"
    )

    # 3) Diagnóstico do roteamento do primeiro estágio
    titulo("3/5 — DIAGNÓSTICO DO ROTEAMENTO")
    print(
        pd.crosstab(
            df["grande_area"],
            df["grande_area_predita_texto2area"],
            margins=True,
        ).to_string()
    )
    for grande in (HUMANAS, LLA):
        mask = df["grande_area"].eq(grande)
        taxa = np.mean(
            df.loc[mask, "grande_area_predita_texto2area"].to_numpy() == grande
        )
        print(f"\nRoteamento correto para {grande}: {taxa:.4%}")

    # 4) Treino dos especialistas
    titulo("4/5 — TREINAMENTO DOS DOIS ESPECIALISTAS")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "seed": SEED,
        "corpus": str(CORPUS),
        "texto2area_model": str(T2A_MODEL),
        "texto2area_vectorizer": str(T2A_VECTORIZER),
        "features": {
            "margens": len(classes_grandes),
            "one_hot_grande_area": len(classes_grandes),
            "top_termos": args.top_termos,
            "hash_features": args.hash_features,
        },
        "humanas": {
            "grande_area": HUMANAS,
            "classes": CLASSES_HUMANAS,
        },
        "lla": {
            "grande_area": LLA,
            "classes": CLASSES_LLA,
        },
        "balanceamento": {
            "metodo": "undersampling_aleatorio",
            "antes_do_split": True,
            "ativo": bool(args.balancear_treino),
        },
        "xgboost": {
            "device": "cuda",
            "tree_method": "hist",
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "min_child_weight": args.min_child_weight,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "max_bin": args.max_bin,
            "early_stopping_rounds": args.early_stopping,
        },
    }
    (OUT_DIR / "configuracao_experimento.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    mask_h = df["grande_area"].eq(HUMANAS).to_numpy()
    mask_lla = df["grande_area"].eq(LLA).to_numpy()

    # O treino de cada cabeça usa o rótulo VERDADEIRO de grande área.
    # Na inferência, a seleção da cabeça será feita pela previsão do texto2area.
    res_h = treinar_especialista(
        "ciencias_humanas",
        X_all[mask_h],
        df.loc[mask_h, "area_classe"].to_numpy(),
        CLASSES_HUMANAS,
        OUT_DIR,
        args,
    )

    res_lla = treinar_especialista(
        "linguistica_letras_artes",
        X_all[mask_lla],
        df.loc[mask_lla, "area_classe"].to_numpy(),
        CLASSES_LLA,
        OUT_DIR,
        args,
    )

    # 5) Resumo
    titulo("5/5 — RESUMO")
    resumo = pd.DataFrame([
        {
            "especialista": "CIÊNCIAS HUMANAS",
            "documentos": res_h["metadata"]["n_documentos"],
            "documentos_balanceados": res_h["metadata"]["n_documentos_balanceados"],
            "treino": res_h["metadata"]["n_treino"],
            "accuracy": res_h["metadata"]["accuracy"],
            "balanced_accuracy": res_h["metadata"]["balanced_accuracy"],
            "f1_macro": res_h["metadata"]["f1_macro"],
            "f1_weighted": res_h["metadata"]["f1_weighted"],
        },
        {
            "especialista": "LINGUÍSTICA, LETRAS E ARTES",
            "documentos": res_lla["metadata"]["n_documentos"],
            "documentos_balanceados": res_lla["metadata"]["n_documentos_balanceados"],
            "treino": res_lla["metadata"]["n_treino"],
            "accuracy": res_lla["metadata"]["accuracy"],
            "balanced_accuracy": res_lla["metadata"]["balanced_accuracy"],
            "f1_macro": res_lla["metadata"]["f1_macro"],
            "f1_weighted": res_lla["metadata"]["f1_weighted"],
        },
    ])
    print(resumo.to_string(index=False))
    resumo.to_csv(
        OUT_DIR / "resumo_experimento.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nTempo total: {time.time() - inicio_total:,.1f} s")
    print(f"Artefatos: {OUT_DIR}")

    if args.texto:
        testar_texto(
            args.texto,
            {
                "ciencias_humanas": {
                    "model": res_h["model"],
                    "classes": CLASSES_HUMANAS,
                },
                "linguistica_letras_artes": {
                    "model": res_lla["model"],
                    "classes": CLASSES_LLA,
                },
            },
            classes_grandes,
            args.hash_features,
        )


if __name__ == "__main__":
    main()
