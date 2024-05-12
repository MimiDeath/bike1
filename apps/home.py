import numpy as np
import streamlit as st
import pandas as pd
from pathlib import Path
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns
import matplotlib.pyplot as plt
from pygwalker.api.streamlit import StreamlitRenderer
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
from plotly.subplots import make_subplots


@st.cache_data
def get_data_info(df):
    info = pd.DataFrame()
    info.index = df.columns
    info['Тип данных'] = df.dtypes
    info['Уникальных'] = df.nunique()
    info['Количество значений'] = df.count()
    return info


@st.cache_data
def get_profile_report(df):
    from pandas_profiling import ProfileReport
    profile = ProfileReport(
        df,
        title="Анализ данных о велосипедных поездках",
        dataset={
            "description": "Этот отчет представляет собой анализ набора данных о городском велосипедном транспорте в Лондоне, который включает информацию о времени, погодных условиях и количестве поездок."
        },
        variables={
            "descriptions": {
                "timestamp": "Временная метка каждой поездки. Используется для анализа активности по времени.",
                "cnt": "Количество новых прокатов велосипедов за интервал. Показатель спроса на велосипедные поездки.",
                "t1": "Реальная температура воздуха в градусах Цельсия во время поездки.",
                "t2": "Ощущаемая температура в градусах Цельсия. Учитывает влияние ветра и влажности.",
                "hum": "Влажность воздуха в процентах во время поездки.",
                "wind_speed": "Скорость ветра в км/ч. Важный фактор для ощущаемой температуры.",
                "weather_code": "Код погодных условий, влияющих на использование велосипедов.",
                "is_holiday": "Индикатор праздничных дней. 1, если день является праздничным, иначе 0.",
                "is_weekend": "Индикатор выходных дней. 1, если день является выходным, иначе 0.",
                "season": "Сезон, в котором происходит поездка. Важно для сезонного анализа спроса."
            }
        }
    )
    return profile


@st.cache_resource
def display_pygwalker(df) -> "StreamlitRenderer":
    vis_spec = r"""{"config":[{"config":{"defaultAggregated":false,"geoms":["boxplot"],"coordSystem":"generic","limit":-1,"timezoneDisplayOffset":0},"encodings":{"dimensions":[{"dragId":"gw_z_UI","fid":"weather_code","name":"weather_code","basename":"weather_code","semanticType":"quantitative","analyticType":"dimension","offset":0},{"dragId":"gw_gnhH","fid":"is_holiday","name":"is_holiday","basename":"is_holiday","semanticType":"quantitative","analyticType":"dimension","offset":0},{"dragId":"gw_p4EU","fid":"is_weekend","name":"is_weekend","basename":"is_weekend","semanticType":"quantitative","analyticType":"dimension","offset":0},{"dragId":"gw_izyg","fid":"season","name":"season","basename":"season","semanticType":"quantitative","analyticType":"dimension","offset":0},{"dragId":"gw_eRlB","fid":"Apr","name":"Apr","basename":"Apr","semanticType":"quantitative","analyticType":"dimension","offset":0},{"dragId":"gw_6Q38","fid":"Aug","name":"Aug","basename":"Aug","semanticType":"quantitative","analyticType":"dimension","offset":0},{"dragId":"gw_JPIf","fid":"Dec","name":"Dec","basename":"Dec","semanticType":"quantitative","analyticType":"dimension","offset":0},{"dragId":"gw_d_Fr","fid":"Feb","name":"Feb","basename":"Feb","semanticType":"quantitative","analyticType":"dimension","offset":0},{"dragId":"gw_WUvw","fid":"Jan","name":"Jan","basename":"Jan","semanticType":"quantitative","analyticType":"dimension","offset":0},{"dragId":"gw_bAV7","fid":"July","name":"July","basename":"July","semanticType":"quantitative","analyticType":"dimension","offset":0},{"dragId":"gw_ORfD","fid":"June","name":"June","basename":"June","semanticType":"quantitative","analyticType":"dimension","offset":0},{"dragId":"gw_nt_G","fid":"Mar","name":"Mar","basename":"Mar","semanticType":"quantitative","analyticType":"dimension","offset":0},{"dragId":"gw_-Whx","fid":"May","name":"May","basename":"May","semanticType":"quantitative","analyticType":"dimension","offset":0},{"dragId":"gw_sFKE","fid":"Nov","name":"Nov","basename":"Nov","semanticType":"quantitative","analyticType":"dimension","offset":0},{"dragId":"gw_KZMp","fid":"Oct","name":"Oct","basename":"Oct","semanticType":"quantitative","analyticType":"dimension","offset":0},{"dragId":"gw_ZGnv","fid":"Sep","name":"Sep","basename":"Sep","semanticType":"quantitative","analyticType":"dimension","offset":0},{"dragId":"gw_jNQW","fid":"Friday","name":"Friday","basename":"Friday","semanticType":"quantitative","analyticType":"dimension","offset":0},{"dragId":"gw_SYZm","fid":"Monday","name":"Monday","basename":"Monday","semanticType":"quantitative","analyticType":"dimension","offset":0},{"dragId":"gw_HwrZ","fid":"Saturday","name":"Saturday","basename":"Saturday","semanticType":"quantitative","analyticType":"dimension","offset":0},{"dragId":"gw_PQxp","fid":"Sunday","name":"Sunday","basename":"Sunday","semanticType":"quantitative","analyticType":"dimension","offset":0},{"dragId":"gw_p5fW","fid":"Thursday","name":"Thursday","basename":"Thursday","semanticType":"quantitative","analyticType":"dimension","offset":0},{"dragId":"gw_DH9K","fid":"Tuesday","name":"Tuesday","basename":"Tuesday","semanticType":"quantitative","analyticType":"dimension","offset":0},{"dragId":"gw_PhpP","fid":"Wednesday","name":"Wednesday","basename":"Wednesday","semanticType":"quantitative","analyticType":"dimension","offset":0},{"dragId":"gw_mea_key_fid","fid":"gw_mea_key_fid","name":"Measure names","analyticType":"dimension","semanticType":"nominal"}],"measures":[{"dragId":"gw_73L8","fid":"cnt","name":"cnt","basename":"cnt","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"dragId":"gw_qi30","fid":"t1","name":"t1","basename":"t1","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"dragId":"gw_cR2-","fid":"t2","name":"t2","basename":"t2","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"dragId":"gw_UbZh","fid":"hum","name":"hum","basename":"hum","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"dragId":"gw_InxO","fid":"wind_speed","name":"wind_speed","basename":"wind_speed","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"dragId":"gw_count_fid","fid":"gw_count_fid","name":"Row count","analyticType":"measure","semanticType":"quantitative","aggName":"sum","computed":true,"expression":{"op":"one","params":[],"as":"gw_count_fid"}},{"dragId":"gw_mea_val_fid","fid":"gw_mea_val_fid","name":"Measure values","analyticType":"measure","semanticType":"quantitative","aggName":"sum"}],"rows":[{"dragId":"gw_CejD","fid":"cnt","name":"cnt","basename":"cnt","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0}],"columns":[{"dragId":"gw_Ngxm","fid":"season","name":"season","basename":"season","semanticType":"quantitative","analyticType":"dimension","offset":0}],"color":[],"opacity":[],"size":[],"shape":[],"radius":[],"theta":[],"longitude":[],"latitude":[],"geoId":[],"details":[],"filters":[{"dragId":"gw_ICsu","fid":"season","name":"season","basename":"season","semanticType":"quantitative","analyticType":"dimension","offset":0,"rule":null}],"text":[]},"layout":{"showActions":false,"showTableSummary":false,"stack":"stack","interactiveScale":false,"zeroScale":true,"size":{"mode":"auto","width":320,"height":200},"format":{},"geoKey":"name","resolve":{"x":false,"y":false,"color":false,"opacity":false,"shape":false,"size":false}},"visId":"gw_kQOm","name":"Chart 1"}],"chart_map":{},"workflow_list":[{"workflow":[{"type":"view","query":[{"op":"raw","fields":["season","cnt"]}]}]}],"version":"0.4.7"}"""
    return StreamlitRenderer(df, spec=vis_spec, dark='light', spec_io_mode="rw")


@st.cache_data
def create_plot(df, feature):
    fig = go.Figure(go.Scatter(
        x=df.index,
        y=df[feature],
        line=dict(width=3)
    ))

    fig.update_layout(
        title=f"Распределение {feature}",
        template="plotly_white",
        xaxis_title='Время',
        yaxis_title=feature,
        font=dict(size=12),
        xaxis=dict(
            tickangle=45,
            type='category',
            ticklabelstep=3
        )
    )
    return fig


@st.cache_data
def create_histogram(df, column_name):
    fig = px.histogram(
        df,
        x=column_name,
        marginal="box",
        title=f"Распределение {column_name}",
        template="plotly"
    )
    return fig


@st.cache_data
def create_correlation_matrix(df, features):
    corr = df[features].corr().round(2)
    fig1 = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        colorscale='ice',
        annotation_text=corr.values
    )
    fig1.update_layout(height=800)

    # Выбираем только корреляцию с целевым признаком 'cnt'
    corr = corr['cnt'].drop('cnt')
    corr = corr[abs(corr).argsort()]
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=corr.values,
        y=corr.index,
        orientation='h',
        marker_color=list(range(len(corr.index))),
    ))
    fig2.update_layout(
        title='Корреляция с cnt',
        height=700,
        xaxis=dict(title='Признак'),  # Название оси x
        yaxis=dict(title='Корреляция'),
    )
    return fig1, fig2


@st.cache_data
def create_correlation_df(df, features, target_feature):
    correlation_matrix = df[features].corr()
    correlation_with_target = correlation_matrix[target_feature].round(2)
    correlation_df = pd.DataFrame({
        'Признак': correlation_with_target.index,
        'Корреляция с ' + target_feature: correlation_with_target.values
    })
    return correlation_df


@st.cache_data
def create_countplot(df, categorical_features):
    sns.set_theme(style="white")
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 35))
    fig.subplots_adjust(hspace=0.5, bottom=0)

    for ax, catplot in zip(axes.flatten(), categorical_features):
        sns.countplot(x=catplot, data=df, ax=ax)
        ax.set_title(catplot.upper(), fontsize=18)
        ax.set_ylabel('Count', fontsize=16)
        ax.set_xlabel(f'{catplot} Values', fontsize=15)
        ax.tick_params(axis='x', rotation=45)
    return fig


@st.cache_data
def create_pairplot(df, selected_features, hue=None):
    sns.set_theme(style="whitegrid")
    pairplot_fig = sns.pairplot(
        df,
        vars=selected_features,
        hue=hue,
        palette='viridis',
        plot_kws={'alpha': 0.5, 's': 80, 'edgecolor': 'k'},
        height=3
    )
    plt.subplots_adjust(top=0.95)
    return pairplot_fig


def display_scatter_plot(df, numerical_features, categorical_features):
    from scipy.stats import stats
    c1, c2, c3, c4 = st.columns(4)
    feature1 = c1.selectbox('Первый признак', numerical_features, key='scatter_feature1')
    feature2 = c2.selectbox('Второй признак', numerical_features + categorical_features, index=4,
                            key='scatter_feature2')
    filter_by = c3.selectbox('Фильтровать по', [None, *categorical_features],
                             key='scatter_filter_by')

    correlation = round(stats.pearsonr(df[feature1], df[feature2])[0], 4)
    c4.metric("Корреляция", correlation)

    fig = px.scatter(
        df,
        x=feature1, y=feature2,
        color=filter_by, trendline='ols',
        opacity=0.5,
        template='plotly',
        title=f'Корреляция между {feature1} и {feature2}'
    )
    st.plotly_chart(fig, use_container_width=True)


def app(df, current_dir: Path):
    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃                     Главная страница                     ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    st.title("Анализ спроса на прокат велосипедов в городе")
    st.markdown("## Область применения")
    markdown_col1, markdown_col2 = st.columns(2)

    markdown_col1.markdown(
        """
        Эта страница предназначена для описания использования данных и общего контекста проекта. 
        Анализ использования городского велосипедного транспорта основывается на обработке и исследовании данных о велосипедных поездках в Лондоне. Эти данные представляют собой ценный источник информации для понимания текущих тенденций и моделей использования велосипедов в городских условиях, а также для планирования будущих улучшений в инфраструктуре и политике в области велосипедного транспорта.
        """
    )
    markdown_col2.image(str(current_dir / 'images' / 'img.png'), width=500, use_column_width='auto')

    tab1, tab2 = st.tabs(["Описание данных", "Пример данных"])

    with tab1:
        st.markdown(
            r"""
            ## Ключевые параметры и характеристики данных
            
            | Параметр     | Описание                                                                                          |
            |--------------|---------------------------------------------------------------------------------------------------|
            | timestamp    | Временная метка данных                                                                            |
            | cnt          | Количество новых прокатов велосипедов                                                            |
            | t1           | Реальная температура в градусах Цельсия                                                          |
            | t2           | Температура в градусах Цельсия, "ощущаемая"                                                       |
            | hum          | Влажность в процентах                                                                             |
            | wind_speed   | Скорость ветра в км/ч                                                                             |
            | weather_code | Категория погоды                                                                                  |
            | is_holiday   | Бинарный признак: является ли день праздничным (1 - праздник, 0 - не праздник)                   |
            | is_weekend   | Бинарный признак: является ли день выходным (1 - выходной, 0 - не выходной)                     |
            | season       | Категориальный признак: метеорологическое время года (0 - весна, 1 - лето, 2 - осень, 3 - зима) |
            </br>
            
            Описание категорий weather_code:
            
            * 1: Ясно; в основном ясно, но могут быть небольшие пятна тумана / дыма / тумана в окрестностях.
            * 2: Рассеянные облака / небольшое количество облаков.
            * 3: Разорванные облака.
            * 4: Облачно.
            * 7: Дождь / небольшой ливень / легкий дождь.
            * 10: Дождь с грозой.
            * 26: Снегопад.
            * 94: Морозный туман.
            Эти параметры будут использоваться для анализа и прогнозирования спроса на велосипеды в зависимости от погодных условий, времени года, а также выявления паттернов использования велосипедов в выходные и во время праздников.
            """,
            unsafe_allow_html=True
        )
    with tab2:
        st.header("Пример данных")
        st.dataframe(df.head(15))

    categorical_features = ['weather_code', 'is_holiday', 'is_weekend', 'season', 'Apr', 'Aug', 'Dec', 'Feb', 'Jan',
                            'July', 'June', 'Mar', 'May', 'Nov', 'Oct', 'Sep', 'Friday', 'Monday', 'Saturday', 'Sunday',
                            'Thursday', 'Tuesday', 'Wednesday']
    numerical_features = [col for col in df.columns if col not in categorical_features]

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃               Предварительный анализ данных                 ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    st.subheader("Предварительный анализ данных")
    # Отображение метрик в колонках
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Средняя температура (°C)", f"{df['t1'].mean():.2f}")
    with col2:
        st.metric("Средняя влажность (%)", f"{df['hum'].mean():.2f}")
    with col3:
        st.metric("Средняя скорость ветра (км/ч)", f"{df['wind_speed'].mean():.2f}")

    st.dataframe(get_data_info(df), use_container_width=True)

    st.markdown("""
    Предварительный анализ данных показал следующее:
    * всего в наборе данных содержится 730 строк и 28 столбцов;
    * не обнаружено пропущенных значений в данных;
    * в столбце timestamp необходимо преобразовать тип данных в формат datetime для более удобной работы с временными данным.
    """)
    st.subheader("Основные статистики для признаков")

    st.subheader("Рассчитаем основные статистики для числовых признаков")
    st.dataframe(df.describe())
    st.markdown("""
        Данные не содержат дубликатов и пропущенных значений, что указывает на их хорошее качество и готовность к анализу. Вот основные статистические характеристики числовых признаков в нашем наборе данных:
        * количество поездок (cnt): среднее значение - 1143 поездок, с минимумом в 0 и максимумом в 7860 поездок. Распределение числа поездок показывает значительную вариативность, отражающую различия в использовании велосипедов в разное время и при разных условиях;
        * температура воздуха (t1): средняя температура составляет 12.47°C, с диапазоном от -1.5°C до 34°C. Это указывает на широкий диапазон погодных условий, в которых происходят поездки;
        * ощущаемая температура (t2): среднее значение близко к реальной температуре воздуха, что указывает на важность учета ощущаемой температуры при анализе использования велосипедов;
        * влажность (hum): средняя влажность составляет 72.32%, что может влиять на комфортность использования велосипедов;
        * скорость ветра (wind_speed): средняя скорость ветра составляет 15.91 км/ч, что также является важным фактором комфорта езды на велосипеде.
        Эти статистические данные помогут нам лучше понять, как различные условия влияют на использование городского велосипедного транспорта в Лондоне. 
    """)

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃                     Интерактивные отчёты                     ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    st.subheader("Анализ данных")
    tab1, tab2 = st.tabs(["Редактор графиков", "Показать отчет о данных"])
    with tab1:
        renderer = display_pygwalker(df)
        renderer.render_explore()
    with tab2:
        if st.button("Сформировать отчёт", type='primary', use_container_width=True):
            st_profile_report(get_profile_report(df))

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃                     Визуализация                     ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    st.header("Визуализация числовых признаков")
    selected_feature = st.selectbox(
        "Выберите признак",
        numerical_features,
        key="create_histogram_selectbox"
    )
    # Построение и отображение графика
    plot_fig = create_plot(
        df,
        selected_feature
    )
    st.plotly_chart(plot_fig, use_container_width=True)
    hist_fig = create_histogram(
        df,
        selected_feature
    )
    st.plotly_chart(hist_fig, use_container_width=True)

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃                     Корреляция                     ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    st.header("Корреляционный анализ")
    st.subheader("Распределение по различным признакам")
    display_scatter_plot(df, numerical_features, categorical_features)

    if st.button('Показать все переменные на корреляционной матрице', use_container_width=True):
        corr_features = numerical_features + categorical_features
    else:
        corr_features = numerical_features

    fig1, fig2 = create_correlation_matrix(df, corr_features)
    st.plotly_chart(fig1, use_container_width=True)

    markdown_col1, markdown_col2 = st.columns(2)
    markdown_col1.markdown("""
        Корреляционная матрица представляет связь между различными числовыми параметрами. В данном случае: 
        1.	количество прокатов велосипедов (cnt) имеет положительную корреляцию с температурой (t1 и t2). Это ожидаемо, так как в более теплую погоду люди склонны больше пользоваться велосипедами;
        2.	столбцы t1 и t2 (реальная температура и температура “ощущается”) сильно коррелируют между собой (коэффициент корреляции близок к 1), что логично, так как это два различных способа измерения температуры;
        3.	влажность (hum) имеет отрицательную корреляцию с количеством прокатов велосипедов и положительную корреляцию с температурой. Это может означать, что в сырую и влажную погоду люди менее склонны к прокату велосипедов;
        4.	скорость ветра (wind_speed) имеет слабую корреляцию с количеством прокатов велосипедов и другими признаками. 
        Из анализа корреляций видно, что количество прокатов велосипедов обычно увеличивается в теплую погоду, а также в условиях с низкой влажностью. Скорость ветра, кажется, не оказывает существенного влияния на количество прокатов. Также заметно, что показатели реальной температуры и температуры “ощущается” сильно коррелируют 
    """)
    markdown_col2.plotly_chart(fig2, use_container_width=True)

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃                     Диаграммы                        ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    st.markdown(
        """
        ## Столбчатые диаграммы для категориальных признаков
        """
    )
    st.pyplot(create_countplot(df, categorical_features))

    st.markdown(
        """
        ## Точечные диаграммы для пар числовых признаков
        """
    )
    selected_features = st.multiselect(
        'Выберите признаки',
        numerical_features + categorical_features,
        default=numerical_features,
        key='pairplot_vars'
    )

    # Опциональный выбор категориальной переменной для цветовой дифференциации
    hue_option = st.selectbox(
        'Выберите признак для цветового кодирования (hue)',
        ['None'] + categorical_features,
        index=1,
        key='pairplot_hue'
    )
    if hue_option == 'None':
        hue_option = None
    if selected_features:
        st.pyplot(create_pairplot(df, selected_features, hue=hue_option))
    else:
        st.error("Пожалуйста, выберите хотя бы один признак для создания pairplot.")
