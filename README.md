# TCC: Systemic AI Approaches for Anomaly Detection

---

## 🇧🇷 Versão em Português

### Sobre o Projeto

Este projeto implementa um sistema de detecção de anomalias em tráfego de rede utilizando técnicas avançadas de Machine Learning e Deep Learning. O sistema é capaz de identificar ataques cibernéticos e comportamentos anômalos em tempo real, alcançando uma precisão superior a 99% na detecção de ameaças.

### Principais Características
- **Detecção Binária**: Classifica tráfego como BENIGNO ou ATAQUE
- **Múltiplos Modelos**: Random Forest, SVM e Isolation Forest
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
1. Crie um ambiente virtual:
   - Windows: `python -m venv venv` e depois `venv\Scripts\activate`
   - Linux/Mac: `python -m venv venv` e depois `source venv/bin/activate`
2. Instale as dependências: `pip install -r requirements.txt`
3. Baixe os datasets (veja seção Datasets abaixo)
4. Execute o pré-processamento: `python src/data_preprocessing.py --dataset cic`
5. Treine os modelos:
   - Teste rápido: `python src/model_training_fast.py`
   - Treinamento completo: `python src/model_training_complete.py`
6. Avalie os resultados: `python src/evaluation.py`
7. Detecte drift: `python src/drift_detection.py`

### Estrutura do Projeto
- `src/`: Scripts principais
  - `data_preprocessing.py`: Pré-processamento de dados
  - `model_training.py`: Treinamento de modelos (versão original)
  - `model_training_fast.py`: Treinamento rápido para testes
  - `model_training_complete.py`: Treinamento de múltiplos modelos
  - `evaluation.py`: Avaliação de modelos
  - `drift_detection.py`: Detecção de drift conceitual
- `notebooks/`: Notebooks Jupyter para exploração
- `data/`: Datasets (baixar separadamente)
- `models/`: Modelos treinados salvos
- `figures/`: Gráficos e visualizações geradas

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
  - `data_preprocessing.py`: Data preprocessing
  - `model_training.py`: Model training (original version)
  - `model_training_fast.py`: Fast training for testing
  - `model_training_complete.py`: Multiple models training
  - `evaluation.py`: Model evaluation
  - `drift_detection.py`: Concept drift detection
- `notebooks/`: Jupyter notebooks for exploration
- `data/`: Datasets (download separately)
- `models/`: Saved trained models
- `figures/`: Generated plots and visualizations

### Datasets
- **CIC-IDS 2017**: https://www.unb.ca/cic/datasets/ids-2017.html (2.8M records)
  - Download CSVs to `data/raw/cic-ids/MachineLearningCSV/`

### Results
- **Random Forest**: F1-Score 99.6%, Precision 99.7%, Recall 99.5%
- **Drift Detection**: Stable system with no significant drift
- Plots available in `figures/` after execution
