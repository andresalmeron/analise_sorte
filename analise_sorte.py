import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import re

st.set_page_config(page_title="Sorte ou Habilidade?", layout="wide")

def ler_arquivo(arquivo):
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
        raise ValueError("O arquivo não possui duas colunas úteis (Data e Cota/Preço).")
    return df

def limpar_numero(val):
    if pd.isna(val): return np.nan
    if isinstance(val, (int, float)): return float(val)
    val_str = str(val).strip()
    val_limpo = re.sub(r'[^\d.,-]', '', val_str)
    if val_limpo == '': return np.nan
    
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

arquivo_upload = st.file_uploader("Faça o upload da planilha (Excel ou CSV da extração com os PREÇOS)", type=['csv', 'xlsx'])

if arquivo_upload is not None:
    df_raw = ler_arquivo(arquivo_upload)

    with st.expander("🛠️ Modo Depuração (Leitura Bruta)"):
        st.dataframe(df_raw.head())

    try:
        df_completo = df_raw.loc[:, ~df_raw.columns.str.contains('^Unnamed')].copy()
        df_completo = df_completo.iloc[:, :2]
        df_completo.columns = ['Data', 'Cota']
        
        df_completo['Data'] = pd.to_datetime(df_completo['Data'], dayfirst=True, errors='coerce')
        df_completo['Cota'] = df_completo['Cota'].apply(limpar_numero)
        
        df_completo = df_completo.dropna(subset=['Data', 'Cota'])
        
        df_completo = df_completo.sort_values('Data').reset_index(drop=True)
        df_completo['Retorno'] = df_completo['Cota'].pct_change()
        
        df_completo = df_completo.replace([np.inf, -np.inf], np.nan).dropna(subset=['Retorno']).reset_index(drop=True)
        
        anos_disponiveis = len(df_completo) / 12
        
        st.divider()
        st.subheader("Filtro de Período")
        
        # ==========================================================
        # O SLIDER DE TEMPO
        # Padrão definido para 10 anos (ou o máximo, se o fundo for novo)
        # ==========================================================
        max_anos_slider = int(np.ceil(anos_disponiveis))
        valor_padrao = min(10, max_anos_slider)
        
        anos_analise = st.slider(
            "Selecione o horizonte de análise (últimos X anos):",
            min_value=1,
            max_value=max_anos_slider,
            value=valor_padrao,
            step=1
        )
        
        # Recorta o dataframe com base na seleção do usuário
        meses_corte = anos_analise * 12
        df = df_completo.tail(meses_corte).reset_index(drop=True)
        
        n_meses = len(df)
        anos = n_meses / 12
        retorno_medio_mensal = df['Retorno'].mean()
        volatilidade_mensal = df['Retorno'].std()
        
        retorno_anualizado = ((1 + retorno_medio_mensal) ** 12) - 1
        volatilidade_anualizada = volatilidade_mensal * np.sqrt(12)
        
        st.subheader("1. Resumo da Amostra (Período Selecionado)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Período Analisado", f"{formatar_decimal_br(anos)} anos")
        col2.metric("Meses (N)", f"{n_meses}")
        col3.metric("Retorno Médio (a.a.)", formatar_percentual_br(retorno_anualizado))
        col4.metric("Volatilidade (a.a.)", formatar_percentual_br(volatilidade_anualizada))
        
        st.divider()

        st.subheader("2. O Multiverso do Azar: Simulação de Monte Carlo")
        st.markdown(f"""
        Se o mercado é um passeio aleatório, o que acontece se pegarmos todos os retornos mensais dos últimos **{anos_analise} anos** e sortearmos a ordem deles 10.000 vezes? 
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
            
            # A Guilhotina Estatística mantém a escala limpa
            teto_simulado = np.percentile(caminhos_acumulados[-1, :], 99)
            teto_real = np.max(trajetoria_real)
            limite_superior = max(teto_simulado, teto_real) * 1.05 
            
            ax.set_ylim(bottom=0, top=limite_superior)
            
            ax.set_title(f'Trajetória Real vs. {num_simulacoes} Caminhos Aleatórios ({anos_analise} anos)', fontsize=14)
            ax.set_ylabel('Fator de Crescimento do Capital')
            ax.set_xlabel('Meses Alocados')
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            ax.legend()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(x).replace(',', '.')))
            
        st.pyplot(fig)
        plt.close(fig) 
            
        st.divider()

        st.subheader("3. A Régua da Dúvida: Estatística T")
        st.markdown("""
        A Estatística T traduz o gráfico acima em números: ela mede se o retorno excedente gerado foi forte o suficiente para romper a barreira do ruído e ser considerado estatisticamente significativo na janela de tempo escolhida.
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
