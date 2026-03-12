import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sorte ou Habilidade?", layout="wide")

def ler_arquivo(arquivo):
    """Função robusta para ler CSVs/Excel e contornar os cabeçalhos do Comdinheiro."""
    if arquivo.name.endswith('.csv'):
        try:
            # Tenta primeiro o padrão vírgula
            df = pd.read_csv(arquivo, sep=',', encoding='utf-8')
            if df.shape[1] < 2: # Se leu como 1 coluna só, tenta ponto-e-vírgula
                arquivo.seek(0)
                df = pd.read_csv(arquivo, sep=';', encoding='latin-1')
        except Exception:
            arquivo.seek(0)
            df = pd.read_csv(arquivo, sep=';', encoding='latin-1')
    else:
        df = pd.read_excel(arquivo)
        
    if df.shape[1] < 2:
        raise ValueError("O arquivo não possui duas colunas separadas para Data e Cota.")
    return df

def processar_dados(df):
    """Limpa a base isolando o ruído das extrações."""
    df = df.iloc[:, :2].copy()
    df.columns = ['Data', 'Cota']
    
    # Ao forçar a conversão de Data, cabeçalhos sujos e rodapés viram 'NaT' (Not a Time)
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
    df = df.dropna(subset=['Data']) 
    
    # Conversão inteligente de valores financeiros
    def limpar_numero(val):
        if pd.isna(val): return np.nan
        if isinstance(val, (int, float)): return float(val)
        val = str(val).strip()
        # Apenas remove os pontos se o valor usar vírgula como decimal (Padrão BR)
        if ',' in val:
            val = val.replace('.', '').replace(',', '.')
        try:
            return float(val)
        except:
            return np.nan
            
    df['Cota'] = df['Cota'].apply(limpar_numero)
    df = df.dropna(subset=['Cota'])
    
    # Ordenação cronológica rigorosa
    df = df.sort_values('Data').reset_index(drop=True)
    df['Retorno'] = df['Cota'].pct_change()
    
    # Remove a primeira linha vazia e quaisquer anomalias matemáticas
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Retorno']).reset_index(drop=True)
    
    if len(df) == 0:
        raise ValueError("Não sobrou nenhum dado válido após a limpeza. Verifique o arquivo.")
        
    return df

st.title("Sorte ou Habilidade? O Teste da Aleatoriedade")
st.markdown("""
Esta ferramenta aplica testes estatísticos rigorosos para responder a uma pergunta fundamental na alocação de capital: **O retorno histórico de um fundo é fruto da genialidade do gestor ou apenas um passeio aleatório (Random Walk) guiado pelo ruído do mercado?**
""")

arquivo_upload = st.file_uploader("Faça o upload da planilha (Excel ou CSV da extração)", type=['csv', 'xlsx'])

if arquivo_upload is not None:
    try:
        df_raw = ler_arquivo(arquivo_upload)
        df = processar_dados(df_raw)
        
        n_meses = len(df)
        anos = n_meses / 12
        retorno_medio_mensal = df['Retorno'].mean()
        volatilidade_mensal = df['Retorno'].std()
        
        retorno_anualizado = ((1 + retorno_medio_mensal) ** 12) - 1
        volatilidade_anualizada = volatilidade_mensal * np.sqrt(12)
        
        st.subheader("1. Resumo da Amostra")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Período Analisado", f"{anos:.1f} anos")
        col2.metric("Meses (N)", f"{n_meses}")
        col3.metric("Retorno Médio (a.a.)", f"{retorno_anualizado:.2%}")
        col4.metric("Volatilidade (a.a.)", f"{volatilidade_anualizada:.2%}")
        
        st.divider()

        st.subheader("2. O Multiverso do Azar: Simulação de Monte Carlo")
        st.markdown("""
        Se o mercado é um passeio aleatório, o que acontece se pegarmos todos os retornos mensais deste fundo e **sortearmos a ordem deles 10.000 vezes**? 
        """)
        
        num_simulacoes = 10000
        retornos_historicos = df['Retorno'].values
        
        with st.spinner('Gerando 10.000 universos paralelos (Bootstrapping)...'):
            indices_aleatorios = np.random.randint(0, n_meses, size=(n_meses, num_simulacoes))
            simulacoes_retornos = retornos_historicos[indices_aleatorios]
            
            # Cálculo com base 1.0 (Ponto Zero)
            caminhos_acumulados = np.cumprod(simulacoes_retornos + 1, axis=0)
            caminhos_acumulados = np.vstack([np.ones(num_simulacoes), caminhos_acumulados]) 
            
            trajetoria_real = np.cumprod(retornos_historicos + 1)
            trajetoria_real = np.insert(trajetoria_real, 0, 1.0)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            eixo_x = np.arange(n_meses + 1)
            
            ax.plot(eixo_x, caminhos_acumulados[:, :500], color='lightgray', alpha=0.15)
            ax.plot(eixo_x, trajetoria_real, color='#004488', linewidth=3, label='Trajetória Real do Fundo')
            
            ax.set_title(f'Trajetória Real vs. {num_simulacoes} Caminhos Aleatórios', fontsize=14)
            ax.set_ylabel('Fator de Crescimento do Capital')
            ax.set_xlabel('Meses Alocados')
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            ax.legend()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
        st.pyplot(fig)
        plt.close(fig) 
        
        st.markdown("""
        **Como interpretar a imagem:**
        * A mancha cinza representa o **domínio da sorte**. Todos esses caminhos contêm exatamente as mesmas taxas de retorno que o fundo teve, apenas embaralhadas.
        * Se a linha azul não foge expressivamente da nuvem cinza, o resultado prático se deve fundamentalmente à exposição sistemática aos fatores de risco.
        """)
            
        st.divider()

        st.subheader("3. A Régua da Dúvida: Estatística T")
        st.markdown("""
        A Estatística T traduz o gráfico acima em números: ela mede se o retorno gerado foi forte o suficiente para romper a barreira do ruído e ser considerado estatisticamente significativo.
        """)
        
        t_stat = retorno_medio_mensal / (volatilidade_mensal / np.sqrt(n_meses))
        anos_necessarios = ((2.0 * volatilidade_mensal) / retorno_medio_mensal)**2 / 12 if retorno_medio_mensal > 0 else float('inf')
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric("T-Stat (Ouro > 2.0)", f"{t_stat:.2f}")
            if t_stat >= 2.0:
                st.success("Resultado Estatisticamente Significativo. Há fortes indícios contra o acaso puro.")
            else:
                st.warning("Resultado NÃO Significativo. O retorno médio não superou o ruído da amostra.")
                
        with c2:
            if retorno_medio_mensal > 0:
                st.metric("Anos exigidos para provar habilidade (T=2.0)", f"{anos_necessarios:.1f} anos")
                st.info("O modelo assume que o Alfa é zero. A volatilidade exige tempo para confirmar consistência.")
        
    except Exception as e:
        st.error(f"Erro ao processar o arquivo. Detalhe técnico: {e}")
