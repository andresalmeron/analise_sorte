import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import re

st.set_page_config(page_title="Sorte ou Habilidade? (Qualificado)", layout="wide")

@st.cache_data(ttl=86400) # Mantém a base em cache por 24 horas para evitar lentidão
def carregar_benchmark_default():
    """Baixa a base do IBRX 100 diretamente do repositório no GitHub."""
    # O link 'raw' aponta diretamente para o conteúdo do CSV
    url_ibrx = "https://raw.githubusercontent.com/andresalmeron/analise_sorte/QUALIFICADO/preco%20-%20ibrx.xlsx%20-%20comdinheiro.csv"
    return pd.read_csv(url_ibrx, sep=',', encoding='utf-8')

def ler_arquivo(arquivo, sep_default=';'):
    if getattr(arquivo, 'name', '').endswith('.csv') or (isinstance(arquivo, str) and arquivo.endswith('.csv')):
        try:
            df = pd.read_csv(arquivo, sep=',', encoding='utf-8')
            if df.shape[1] < 2:
                if hasattr(arquivo, 'seek'): arquivo.seek(0)
                df = pd.read_csv(arquivo, sep=sep_default, encoding='latin-1')
        except Exception:
            if hasattr(arquivo, 'seek'): arquivo.seek(0)
            df = pd.read_csv(arquivo, sep=sep_default, encoding='latin-1')
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

st.title("Sorte ou Habilidade? Simulação de Monte Carlo Qualificada")
st.markdown("""
Esta ferramenta aplica um **Monte Carlo Baseado em Fatores** para responder a uma pergunta fundamental na alocação de capital: 
**O retorno histórico de um fundo é fruto de habilidade extraordinária do gestor ou apenas uma exposição sistemática ao risco de mercado (Beta) somada a ruído aleatório?**
""")

col_up1, col_up2 = st.columns(2)
with col_up1:
    arquivo_upload = st.file_uploader("1. Upload da Cota do Fundo (Excel/CSV)", type=['csv', 'xlsx'])

with col_up2:
    arquivo_bench = st.file_uploader("2. Upload do Benchmark Customizado (Opcional)", type=['csv', 'xlsx'])
    st.caption("Se deixado em branco, usaremos o IBRX 100 padrão (via nuvem).")

if arquivo_upload is not None:
    # --- Processamento do Fundo ---
    df_fundo = ler_arquivo(arquivo_upload)
    df_fundo = df_fundo.loc[:, ~df_fundo.columns.str.contains('^Unnamed')].copy().iloc[:, :2]
    df_fundo.columns = ['Data', 'Cota']
    df_fundo['Data'] = pd.to_datetime(df_fundo['Data'], dayfirst=True, errors='coerce')
    df_fundo['Cota'] = df_fundo['Cota'].apply(limpar_numero)
    df_fundo = df_fundo.dropna(subset=['Data', 'Cota']).sort_values('Data').reset_index(drop=True)
    df_fundo['Retorno_Fundo'] = df_fundo['Cota'].pct_change()

    # --- Processamento do Benchmark ---
    if arquivo_bench is not None:
        df_bench_raw = ler_arquivo(arquivo_bench)
    else:
        try:
            df_bench_raw = carregar_benchmark_default()
        except Exception as e:
            st.error(f"⚠️ Erro ao acessar a base do IBRX no GitHub. Verifique sua conexão ou se o arquivo existe no repositório.\nDetalhes técnicos: {e}")
            st.stop()
    
    df_bench = df_bench_raw.loc[:, ~df_bench_raw.columns.str.contains('^Unnamed')].copy().iloc[:, :2]
    df_bench.columns = ['Data', 'Preco_Bench']
    df_bench['Data'] = pd.to_datetime(df_bench['Data'], dayfirst=True, errors='coerce')
    df_bench['Preco_Bench'] = df_bench['Preco_Bench'].apply(limpar_numero)
    df_bench = df_bench.dropna(subset=['Data', 'Preco_Bench']).sort_values('Data').reset_index(drop=True)
    df_bench['Retorno_Bench'] = df_bench['Preco_Bench'].pct_change()

    # --- Merge dos Dados ---
    # É obrigatório que os retornos estejam alinhados pela exata mesma data
    df_completo = pd.merge(df_fundo[['Data', 'Retorno_Fundo']], df_bench[['Data', 'Retorno_Bench']], on='Data', how='inner')
    df_completo = df_completo.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    
    if len(df_completo) < 3:
        st.error("⚠️ Não há dados sobrepostos suficientes entre o Fundo e o Benchmark para rodar a regressão.")
        st.stop()
        
    df_completo['MesAno_Str'] = df_completo['Data'].dt.strftime('%m/%Y')
    opcoes_datas = df_completo['MesAno_Str'].drop_duplicates().tolist()
    
    st.divider()
    st.subheader("Filtro de Período Alinhado")
    
    # --- UX: Seletores de Data ---
    assinatura_arquivo = arquivo_upload.name + str(arquivo_upload.size)
    if "file_hash" not in st.session_state or st.session_state.file_hash != assinatura_arquivo:
        st.session_state.file_hash = assinatura_arquivo
        st.session_state.end_select = opcoes_datas[-1]
        st.session_state.start_select = opcoes_datas[max(0, len(opcoes_datas) - 121)]
        st.session_state.shortcut_select = "Últimos 10 anos"

    def update_shortcut():
        s = st.session_state.shortcut_select
        end_i = opcoes_datas.index(st.session_state.end_select)
        new_start_i = end_i
        if s == "Último 1 ano": new_start_i = max(0, end_i - 12)
        elif s == "Últimos 3 anos": new_start_i = max(0, end_i - 36)
        elif s == "Últimos 5 anos": new_start_i = max(0, end_i - 60)
        elif s == "Últimos 10 anos": new_start_i = max(0, end_i - 120)
        elif s == "Desde o Início": new_start_i = 0
        if s != "Personalizado": st.session_state.start_select = opcoes_datas[new_start_i]

    def update_dates():
        st.session_state.shortcut_select = "Personalizado"

    col1, col2, col3 = st.columns(3)
    with col1: st.selectbox("1. Data Inicial:", options=opcoes_datas, key="start_select", on_change=update_dates)
    with col2: st.selectbox("2. Data Final:", options=opcoes_datas, key="end_select", on_change=update_dates)
    with col3:
        opcoes_atalho = ["Personalizado", "Último 1 ano", "Últimos 3 anos", "Últimos 5 anos", "Últimos 10 anos", "Desde o Início"]
        st.selectbox("3. Janela Rápida:", options=opcoes_atalho, key="shortcut_select", on_change=update_shortcut)
        
    data_inicio_str = st.session_state.start_select
    data_fim_str = st.session_state.end_select
    idx_ini_real = opcoes_datas.index(data_inicio_str)
    idx_fim_real = opcoes_datas.index(data_fim_str)
    
    if idx_ini_real >= idx_fim_real:
        st.error("⚠️ A Data Inicial deve ser obrigatoriamente anterior à Data Final.")
        st.stop()
    
    data_inicio = df_completo[df_completo['MesAno_Str'] == data_inicio_str]['Data'].min()
    data_fim = df_completo[df_completo['MesAno_Str'] == data_fim_str]['Data'].max()
    mask = (df_completo['Data'] >= data_inicio) & (df_completo['Data'] <= data_fim)
    df = df_completo.loc[mask].reset_index(drop=True)
    
    if len(df) < 3:
        st.error("⚠️ O período selecionado precisa ter dados suficientes para rodar os fatores de risco (Mínimo de 3 meses).")
        st.stop()

    n_meses = len(df)
    anos = n_meses / 12
    
    # --- Regressão Linear: Fundo vs Mercado ---
    ret_fundo = df['Retorno_Fundo'].values
    ret_bench = df['Retorno_Bench'].values
    
    # Extração empírica dos fatores da amostra
    beta, alpha = np.polyfit(ret_bench, ret_fundo, 1)
    residuos = ret_fundo - (alpha + beta * ret_bench)
    
    retorno_medio_fundo = np.mean(ret_fundo)
    volatilidade_fundo = np.std(ret_fundo)
    retorno_anual_fundo = ((1 + retorno_medio_fundo) ** 12) - 1
    vol_anual_fundo = volatilidade_fundo * np.sqrt(12)

    st.subheader("1. Resumo da Amostra & Parâmetros do Modelo OLS")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Período Analisado", f"{formatar_decimal_br(anos)} anos")
    col2.metric("Retorno Fundo (a.a.)", formatar_percentual_br(retorno_anual_fundo))
    col3.metric("Volatilidade (a.a.)", formatar_percentual_br(vol_anual_fundo))
    col4.metric("Beta (Exposição)", formatar_decimal_br(beta))
    col5.metric("Alpha Mensal (Descartado)", formatar_percentual_br(alpha))
    
    st.divider()

    # --- Monte Carlo Qualificado ---
    st.subheader("2. Monte Carlo Qualificado (EMH + Beta)")
    st.markdown(f"""
    Simulamos **10.000 cenários paralelos** onde o retorno do fundo é reconstruído através da sua sensibilidade ao mercado (**Beta = {formatar_decimal_br(beta)}**) 
    somada aos seus resíduos específicos, sorteados de forma independente. 
    **Nota:** O Alfa histórico foi descartado na geração dos caminhos. Isso garante que a simulação represente o limite de um fundo estritamente gerido pelo fator Beta.
    """)
    
    num_simulacoes = 10000
    
    with st.spinner('Gerando 10.000 universos governados pelo Beta e pela variância idiossincrática...'):
        idx_mkt = np.random.randint(0, n_meses, size=(n_meses, num_simulacoes))
        idx_res = np.random.randint(0, n_meses, size=(n_meses, num_simulacoes))
        
        sim_mkt = ret_bench[idx_mkt]
        sim_res = residuos[idx_res]
        
        simulacoes_retornos = beta * sim_mkt + sim_res
        
        caminhos_acumulados = np.cumprod(simulacoes_retornos + 1, axis=0)
        caminhos_acumulados = np.vstack([np.ones(num_simulacoes), caminhos_acumulados])
        
        trajetoria_media = np.mean(caminhos_acumulados, axis=1)
        trajetoria_mediana = np.median(caminhos_acumulados, axis=1)
        
        limite_percentil = np.percentile(caminhos_acumulados[-1, :], 95.45)
        mascara_qualificados = caminhos_acumulados[-1, :] <= limite_percentil
        caminhos_qualificados = caminhos_acumulados[:, mascara_qualificados]
        trajetoria_media_qualificada = np.mean(caminhos_qualificados, axis=1)
        
        trajetoria_real = np.cumprod(ret_fundo + 1)
        trajetoria_real = np.insert(trajetoria_real, 0, 1.0)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        eixo_x = np.arange(n_meses + 1)
        
        ax.plot(eixo_x, caminhos_acumulados[:, :500], color='lightgray', alpha=0.15)
        ax.plot(eixo_x, trajetoria_media, color='#E67E22', linestyle='--', linewidth=2.5, label='Média Bruta (Simulação - Distorcida)')
        ax.plot(eixo_x, trajetoria_media_qualificada, color='#27AE60', linestyle='-.', linewidth=2.5, label='Média Qualificada (95,45% Justo)')
        ax.plot(eixo_x, trajetoria_mediana, color='#C0392B', linestyle=':', linewidth=2.5, label='Mediana Simulação (P50 Sistemático)')
        ax.plot(eixo_x, trajetoria_real, color='#004488', linewidth=3, label='Trajetória Real (Gestor)')
        
        teto_simulado = np.percentile(caminhos_acumulados[-1, :], 99)
        teto_real = np.max(trajetoria_real)
        limite_superior = max(teto_simulado, teto_real) * 1.05 
        
        ax.set_ylim(bottom=0, top=limite_superior)
        ax.set_title(f'Trajetória Real vs. {num_simulacoes} Universos (Beta Sistêmico + Resíduo)', fontsize=14)
        ax.set_ylabel('Fator de Crescimento do Capital')
        ax.set_xlabel('Meses Alocados')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(x).replace(',', '.')))
        
    st.pyplot(fig)
    plt.close(fig)
    
    st.markdown("""
    **Interpretação Baseada na EMH:**
    * Como assumimos que o gestor opera inicialmente num mercado eficiente (zeramos o Alfa artificialmente), a simulação mostra o que aconteceria se ele fosse "apenas" um robô que roda no nível de Beta dele. 
    * Se a linha azul espessa (**Trajetória Real**) se descolar persistentemente acima das linhas tracejadas, é o sinal mais claro e qualificado de que esse gestor tem habilidade legítima, não explicada pelo sobe-e-desce da maré.
    """)
