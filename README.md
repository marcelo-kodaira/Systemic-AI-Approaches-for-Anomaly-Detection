# TCC: Systemic AI Approaches for Anomaly Detection

---

## 🇧🇷 Versão em Português

### Sobre o Projeto

Este projeto implementa um sistema de detecção de anomalias em tráfego de rede utilizando técnicas avançadas de Machine Learning e Deep Learning. O sistema é capaz de identificar ataques cibernéticos e comportamentos anômalos em tempo real, alcançando uma precisão superior a 99% na detecção de ameaças.

### Principais Características
- **Detecção Binária**: Classifica tráfego como BENIGNO ou ATAQUE
- **Múltiplos Modelos**: Random Forest, SVM, Isolation Forest, LSTM e CNN-LSTM
- **Alta Performance**: F1-Score > 0.99 no dataset CIC-IDS 2017
- **Otimização Automática**: Utiliza Optuna para ajuste de hiperparâmetros
- **Datasets Públicos**: CIC-IDS 2017 (2.8M registros)

### Como Clonar o Repositório

```bash
# Clone o repositório
git clone https://github.com/MarceloKodaira/Systemic-AI-Approaches-for-Anomaly-Detection.git

# Entre no diretório do projeto
cd Systemic-AI-Approaches-for-Anomaly-Detection
```

### Configuração
0. Crie a pasta "processed" dentro da pasta `\data`
1. Crie a pasta "figures" dentro da pasta pai `\Systemic-AI-Approaches-for-Anomaly-Detection`
2. Crie a pasta "models" dentro da pasta pai `\Systemic-AI-Approaches-for-Anomaly-Detection`
2. Crie um ambiente virtual:
   - Windows: `python -m venv venv` e depois `venv\Scripts\activate`
   - Linux/Mac: `python -m venv venv` e depois `source venv/bin/activate`
3. Instale as dependências: `pip install -r requirements.txt`
4. Baixe os datasets (veja seção Datasets abaixo)
5. Execute o pré-processamento: `python src/data_preprocessing.py --dataset cic`
6. Treine os modelos:
   - Teste rápido: `python src/model_training_fast.py`
   - Treinamento completo: `python src/model_training_complete.py`
7. Avalie os resultados: `python src/evaluation.py`
8. Detecte drift: `python src/drift_detection.py`

### Estrutura do Projeto
- `src/`: Scripts principais
  - `data_preprocessing.py`: Pré-processamento e limpeza dos dados do CIC-IDS 2017
    - Remove valores nulos e infinitos
    - Normaliza features numéricas para [0,1]
    - Converte labels multiclasse para binário (BENIGN=0, ATTACK=1)
    - Divide dados em 70% treino, 15% validação, 15% teste

  - `model_training.py`: Treinamento original com todos os modelos
    - Random Forest, SVM, Isolation Forest, LSTM, CNN-LSTM
    - 30 trials de otimização com Optuna
    - Versão completa mas computacionalmente pesada

  - `model_training_fast.py`: Versão otimizada para testes rápidos
    - Usa apenas 10% dos dados para otimização
    - 3 trials de Optuna em vez de 30
    - Treina apenas Random Forest
    - Executa em 2-3 minutos

  - `model_training_complete.py`: Treina RF, SVM e Isolation Forest
    - Versão intermediária com múltiplos modelos
    - SVM limitado a 10k amostras por performance
    - 5 trials de otimização por modelo

  - `model_training_all.py`: Versão mais eficiente com SGD
    - Substitui SVM por SGDClassifier (100x mais rápido)
    - Treina RF, SGD e Isolation Forest
    - Gera relatório comparativo automático

  - `evaluation.py`: Avaliação completa com bootstrap
    - Calcula intervalos de confiança 95%
    - Cross-validation com 5 folds
    - Gera matriz de confusão e curva ROC
    - ATENÇÃO: Muito lento com datasets grandes

  - `evaluation_fast.py`: Avaliação otimizada sem bootstrap
    - Carrega automaticamente modelos disponíveis
    - Gera gráficos comparativos
    - Salva resultados em CSV
    - Executa em segundos

  - `test_model.py`: Teste rápido de performance
    - Usa apenas 10% do test set
    - Avalia um modelo por vez
    - Resultados em menos de 1 minuto

  - `drift_detection.py`: Detecção de mudanças na distribuição
    - Teste Kolmogorov-Smirnov para drift estatístico
    - Population Stability Index (PSI)
    - Gera visualizações e relatório de drift
    - Simula 5 janelas temporais

- `notebooks/`: Notebooks Jupyter para exploração
- `data/`: Datasets (baixar separadamente)
  - `raw/`: Dados originais do CIC-IDS
  - `processed/`: Dados pré-processados em formato .pkl
- `models/`: Modelos treinados salvos (.pkl)
- `figures/`: Gráficos e visualizações geradas
  - Matriz de confusão
  - Gráficos de drift
  - Comparação de modelos

### Datasets
- **CIC-IDS 2017**: https://www.unb.ca/cic/datasets/ids-2017.html (2.8M registros)
  - Baixe os CSVs para `data/raw/cic-ids/MachineLearningCSV/`

### Resultados
- **Random Forest**: F1-Score de 99.6%, Precisão 99.7%, Recall 99.5%
- **Detecção de Drift**: Sistema estável sem drift significativo
- Gráficos disponíveis em `figures/` após execução

---

## 🇺🇸 English Version

### About the Project

This project implements a network traffic anomaly detection system using advanced Machine Learning and Deep Learning techniques. The system can identify cyber attacks and anomalous behaviors in real-time, achieving over 99% accuracy in threat detection.

### Key Features
- **Binary Detection**: Classifies traffic as BENIGN or ATTACK
- **Multiple Models**: Random Forest, SVM, Isolation Forest, LSTM, and CNN-LSTM
- **High Performance**: F1-Score > 0.99 on CIC-IDS 2017 dataset
- **Automatic Optimization**: Uses Optuna for hyperparameter tuning
- **Public Datasets**: CIC-IDS 2017 (2.8M records)

### How to Clone the Repository

```bash
# Clone the repository
git clone https://github.com/MarceloKodaira/Systemic-AI-Approaches-for-Anomaly-Detection.git

# Enter the project directory
cd Systemic-AI-Approaches-for-Anomaly-Detection
```

### Setup
1. Create a virtual environment:
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - Linux/Mac: `python -m venv venv` then `source venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Download datasets (see Datasets section below)
4. Run preprocessing: `python src/data_preprocessing.py --dataset cic`
5. Train models:
   - Quick test: `python src/model_training_fast.py`
   - Full training: `python src/model_training_complete.py`
6. Evaluate results: `python src/evaluation.py`
7. Detect drift: `python src/drift_detection.py`

### Project Structure
- `src/`: Main scripts
  - `data_preprocessing.py`: CIC-IDS 2017 data preprocessing and cleaning
    - Removes null and infinite values
    - Normalizes numeric features to [0,1]
    - Converts multiclass labels to binary (BENIGN=0, ATTACK=1)
    - Splits data into 70% train, 15% validation, 15% test

  - `model_training.py`: Original training with all models
    - Random Forest, SVM, Isolation Forest, LSTM, CNN-LSTM
    - 30 Optuna optimization trials
    - Complete but computationally heavy version

  - `model_training_fast.py`: Optimized version for quick tests
    - Uses only 10% of data for optimization
    - 3 Optuna trials instead of 30
    - Trains Random Forest only
    - Runs in 2-3 minutes

  - `model_training_complete.py`: Trains RF, SVM, and Isolation Forest
    - Intermediate version with multiple models
    - SVM limited to 10k samples for performance
    - 5 optimization trials per model

  - `model_training_all.py`: Most efficient version with SGD
    - Replaces SVM with SGDClassifier (100x faster)
    - Trains RF, SGD, and Isolation Forest
    - Generates automatic comparative report

  - `evaluation.py`: Complete evaluation with bootstrap
    - Calculates 95% confidence intervals
    - 5-fold cross-validation
    - Generates confusion matrix and ROC curve
    - WARNING: Very slow with large datasets

  - `evaluation_fast.py`: Optimized evaluation without bootstrap
    - Automatically loads available models
    - Generates comparative plots
    - Saves results to CSV
    - Runs in seconds

  - `test_model.py`: Quick performance test
    - Uses only 10% of test set
    - Evaluates one model at a time
    - Results in less than 1 minute

  - `drift_detection.py`: Distribution change detection
    - Kolmogorov-Smirnov test for statistical drift
    - Population Stability Index (PSI)
    - Generates drift visualizations and report
    - Simulates 5 temporal windows

- `notebooks/`: Jupyter notebooks for exploration
- `data/`: Datasets (download separately)
  - `raw/`: Original CIC-IDS data
  - `processed/`: Preprocessed data in .pkl format
- `models/`: Saved trained models (.pkl)
- `figures/`: Generated plots and visualizations
  - Confusion matrices
  - Drift plots
  - Model comparisons

### Datasets
- **CIC-IDS 2017**: https://www.unb.ca/cic/datasets/ids-2017.html (2.8M records)
  - Download CSVs to `data/raw/cic-ids/MachineLearningCSV/`

### Results
- **Random Forest**: F1-Score 99.6%, Precision 99.7%, Recall 99.5%
- **Drift Detection**: Stable system with no significant drift
- Plots available in `figures/` after execution
