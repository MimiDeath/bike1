import sys

import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from pathlib import Path
from apps import home, prediction
from rich.traceback import install
install(show_locals=True)
from rich.console import Console
console = Console()

# Конфигурация страницы Streamlit
st.set_page_config(
    page_title="Прогнозирование использования велосипедов",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)

    # Преобразование к типу даты
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index("timestamp")

    # Добавление колонок
    df["month"] = df.index.month
    df['weekday'] = df.index.day_name()

    # Столбцы, которые нужно преобразовать в int
    columns_to_convert = [
        'weather_code', 'is_holiday', 'is_weekend', 'season'
    ]
    # Преобразование указанных столбцов к типу int
    df[columns_to_convert] = df[columns_to_convert].astype(int)

    # Словарь для замены числовых значений месяцев на строковые
    month_dict = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'June',
        7: 'July', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
    df['month'] = df['month'].map(month_dict)
    df[columns_to_convert] = df[columns_to_convert].astype(int)

    months = pd.get_dummies(df.month)
    months = months.astype(int)
    weekdays = pd.get_dummies(df.weekday)
    weekdays = weekdays.astype(int)

    df = pd.concat([df, months, weekdays], axis=1)

    df.drop(['month', 'weekday'], axis=1, inplace=True)

    int_columns = df.select_dtypes(include='int').columns

    df = df.resample('d').mean().sort_index()
    df = df.dropna()
    df[int_columns] = df[int_columns].astype(int)
    return df


class Menu:
    style = {
        "nav-link": {
            "font-family": "Monospace, Arial",
            "--hover-color": "SkyBlue",
        },
        "nav-link-selected": {
            "background-color": "rgb(10, 0, 124)",
            "font-family": "Monospace , Arial"
        },
    }
    apps = [
        {
            "func"  : home.app,
            "title" : "Главная",
            "icon"  : "house-fill"
        },
        {
            "func"  : prediction.app,
            "title" : "Прогнозирование",
            "icon"  : "lightning-fill"
        },
    ]

    def run(self):
        with st.sidebar:
            titles = [app["title"] for app in self.apps]
            icons  = [app["icon"]  for app in self.apps]
            st.image('images/logo.bmp')

            selected = option_menu(
                "Меню",
                options=titles,
                icons=icons,
                menu_icon="cast",
                styles=self.style,
                default_index=0,
            )

            st.info("""
                ## Прогнозирование использования городского велосипедного транспорта
                Это веб-приложение предназначено для анализа данных о велосипедных поездках в Лондоне.
                Оно помогает понять, как различные факторы влияют на использование городских велосипедов и позволяет прогнозировать будущий спрос.
            """)
        return selected


if __name__ == '__main__':
    try:
        current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
        df = load_data(current_dir / 'london.csv')

        menu = Menu()
        selected = menu.run()

        # Добавление интерфейса для загрузки файла
        st.sidebar.header('Загрузите свой файл')
        uploaded_file = st.sidebar.file_uploader("Выберите CSV файл", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        elif df is None:
            st.sidebar.warning("Пожалуйста, загрузите файл данных.")

        for app in menu.apps:
            if app["title"] == selected:
                app["func"](df, current_dir)
                break

    except Exception as e:
        console.print_exception()
        st.error("Произошла ошибка во время исполнения приложения.", icon="🚨")
        raise


