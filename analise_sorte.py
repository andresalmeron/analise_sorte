import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import re

st.set_page_config(page_title="Sorte ou Habilidade?", layout="wide")

def ler_arquivo(arquivo):
    """Lê o arquivo de forma bruta."""
    if arquivo.name.endswith('.csv'):
        try:
            df = pd.read_csv(arquivo, sep=',', encoding='utf-8')
            if df.shape[1] < 2: 
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

def limpar_numero_trator(val):
    """Ignora o Excel e arranca os números na força bruta com Regex."""
    if pd.isna(val): return np.nan
    if isinstance(val, (int, float)): return float(val)
    
    # Remove qualquer coisa que não seja dígito, ponto, vírgula ou sinal negativo
    val_str = str(val).strip()
    val_limpo = re.sub(r'[^\d.,-]', '', val_str)
    
    if val_limpo == '': return np.nan
    
    # Resolve a treta do decimal
    if ',' in val_limpo and '.' in val_limpo:
        if val_limpo.rfind(',') > val_limpo.rfind('.'):
            val_limpo = val_limpo.replace('.', '').replace(',', '.')
        else:
            val_limpo = val_limpo.replace(',', '')
    elif ',' in val_limpo:
        val_limpo = val_limpo.replace(',', '.')
        
    try:
        return float(val_limpo)
    except:
        return np.nan

def formatar_percentual_br(valor):
    return f"{valor * 100:.2f}%".replace('.', ',')

def formatar_decimal_br(valor):
    return f"{valor:.2f}".replace('.', ',')

st.title("Sorte ou Habilidade? O Teste da Aleatoriedade")
st.markdown("""
Esta ferramenta aplica testes estatísticos rigorosos para responder a uma pergunta fundamental na alocação de capital: **O retorno histórico de um fundo é fruto da genialidade do gestor ou apenas um passeio aleatório guiado pelo ruído do mercado?**
""")

arquivo_upload = st.file_uploader("Faça o upload da planilha (Excel ou CSV da extração)", type=['csv', 'xlsx'])

if arquivo_upload is not None:
    # 1. Leitura Bruta
    df_raw = ler_arquivo(arquivo_upload)
    
    # 2. ABA DE DEPURAÇÃO: Mostra pro André o que diabos o Pandas leu
    with st.expander("🛠️ Modo Depuração (Clique aqui para ver a leitura crua da base)"):
        st.write("Visão das primeiras linhas antes de qualquer limpeza:")
        st.dataframe(df_raw.head())
        st.write(f"Total de linhas importadas: {len(df_raw)}")

    try:
        # 3. Isolamento e Limpeza
        df = df_raw.iloc[:, :2].copy()
        df.columns = ['Data', 'Cota']
        
        # O dayfirst=True força o Pandas a aceitar o padrão BR (DD/MM/AAAA)
        df['Data'] = pd.to_datetime(df['Data'], dayfirst=True, errors='coerce')
        df['Cota'] = df['Cota'].apply(limpar_numero_trator)
        
        linhas_originais = len(df)
        df = df.dropna(subset=['Data', 'Cota'])
        
        if len(df) == 0:
            st.error(f"Puta que pariu! As {linhas_originais} linhas viraram poeira. O Pandas não conseguiu reconhecer nem as datas nem os números. Abra o 'Modo Depuração' acima para ver como os dados vieram corrompidos.")
            st.stop()
            
        # 4. Cálculo
        df = df.sort_values('Data').reset_index(drop=True)
        df['Cota_Base100'] = df['Cota'] / 100.0
        df['Retorno'] = df['Cota_Base100'].pct_change()
        
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Retorno']).reset_index(drop=True)
        
        # 5. Interface Gráfica
        n_meses = len(df)
        anos = n_meses / 12
        retorno_medio_mensal = df['Retorno'].mean()
        volatilidade_mensal = df['Retorno'].std()
        
        retorno_anualizado = ((1 + retorno_medio_mensal) ** 12) - 1
        volatilidade_anualizada = volatilidade_mensal * np.sqrt(12)
        
        st.subheader("1. Resumo da Amostra")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Período Analisado", f"{formatar_decimal_br(anos)} anos")
        col2.metric("Meses (N)", f"{n_meses}")
        col3.metric("Retorno Médio (a.a.)", formatar_percentual_br(retorno_anualizado))
        col4.metric("Volatilidade (a.a.)", formatar_percentual_br(volatilidade_anualizada))
        
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
            
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)).replace(',', '.')))
            
        st.pyplot(fig)
        plt.close(fig) 
            
        st.divider()

        st.subheader("3. A Régua da Dúvida: Estatística T")
        st.markdown("""
        A Estatística T traduz o gráfico acima em números: ela mede se o retorno excedente gerado foi forte o suficiente para romper a barreira do ruído e ser considerado estatisticamente significativo.
        """)
        
        if volatilidade_mensal > 0:
            t_stat = retorno_medio_mensal / (volatilidade_mensal / np.sqrt(n_meses))
            anos_necessarios = ((2.0 * volatilidade_mensal) / retorno_medio_mensal)**2 / 12 if retorno_medio_mensal > 0 else float('inf')
        else:
            t_stat = 0
            anos_necessarios = float('inf')
            
        c1, c2 = st.columns(2)
        with c1:
            st.metric("T-Stat (Ouro > 2,0)", formatar_decimal_br(t_stat))
            if t_stat >= 2.0:
                st.success("Resultado Estatisticamente Significativo. Há indícios matemáticos de skill.")
            else:
                st.warning("Resultado NÃO Significativo. O retorno médio não rompeu o ruído de mercado.")
                
        with c2:
            if retorno_medio_mensal > 0:
                st.metric("Anos exigidos para provar habilidade (T=2,0)", f"{formatar_decimal_br(anos_necessarios)} anos")
                st.info("O modelo assume Alfa zero. Quanto maior a volatilidade, maior o tempo exigido para descartar a sorte.")
        
    except Exception as e:
        st.error(f"Erro fatal: {e}")
