import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import pearsonr

def load_data(file_path):
    data = pd.read_csv(file_path, sep=",")
    return data

def fit_linear_model(data, response_var):
    y = data[response_var]
    X = data.drop(columns=[response_var])
    X = sm.add_constant(X) #добавление константы
    model = sm.OLS(y, X).fit() #инициализация объекта линейной регрессии
    return model


def select_significant_factors(data, response_var):
    y = data[response_var]

    correlation_with_response = {}

    for col in data.columns:
        if col != response_var:
            corr, p_value = pearsonr(data[col], y)
            correlation_with_response[col] = (corr, p_value)
            print(f"{col}: корреляция с откликом = {corr:.4f}, p-value = {p_value:.4f}")

    print("\nСвязь между факторами (коэффициенты корреляции):")
    factors = [col for col in data.columns if col != response_var]
    corr_matrix = data[factors].corr()
    print(corr_matrix)

    print("\nФакторы с высокой корреляцией (|коэффициент корреляции| > 0.8):")
    for i, col1 in enumerate(factors):
        for col2 in factors[i + 1:]:
            corr = corr_matrix.loc[col1, col2]
            if abs(corr) > 0.8:
                print(f"{col1} и {col2}: корреляция = {corr:.4f}")

    print("\nВведите названия факторов, которые хотите оставить (через запятую):")
    selected_factors = input().strip().split(',')
    selected_factors = [factor.strip() for factor in selected_factors if factor in correlation_with_response]

    return selected_factors

def evaluate_model(model, data, ignificance_level):
    r_squared = model.rsquared
    f_statistic = model.fvalue
    f_pvalue = model.f_pvalue
    y_true = data.iloc[:, 0]
    y_pred = model.predict(sm.add_constant(data.iloc[:, 1:]))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    print(f"F-статистика: {f_statistic:.4f}, p-value: {f_pvalue:.4f}")
    if f_pvalue < ignificance_level:
        print("Модель адекватна.")
    else:
        print("Модель неадекватна.")

    print(f"Коэффициент детерминации (R-squared): {r_squared:.4f}")
    print(f"RMSE: {rmse:.4f}")

    return r_squared, rmse

def make_predictions(model, new_data_file, selected_factors):
    new_data = pd.read_csv(new_data_file, sep=",")
    new_data = new_data[selected_factors]
    new_data = sm.add_constant(new_data)
    predictions = model.predict(new_data)
    print("\nПредсказание на новых данных:")
    print(predictions)
    return predictions


def save_results(output_file, content):
    with open(output_file, 'w') as f:
        f.write(content)

# Основная функция
def main():
    data_file = 'data.txt' #файл для обучения
    response_var = 'y'
    new_data_file = 'new_data.txt' #новые значения факторов
    output_file = 'output.txt'

    significance_level = float(input("Введите уровень значимости (например, 0.05): ").strip())

    # Загрузка данных из файла
    data = load_data(data_file)

    # Оценка параметров модели
    initial_model = fit_linear_model(data, response_var)

    # Отбор значимых факторов
    selected_factors = select_significant_factors(data, response_var)

    data_selected = data[[response_var] + selected_factors]

    model_selected = fit_linear_model(data_selected, response_var)

    # Оценка адекватности модели
    r_squared, rmse = evaluate_model(model_selected, data_selected, significance_level)

    # Выполнение предсказания
    predictions = make_predictions(model_selected, new_data_file, selected_factors)

    # Формирование текстового вывода
    output_content = (
        f"Выбранные факторы: {selected_factors}\n"
        f"R-squared: {r_squared}\n"
        f"RMSE: {rmse}\n"
        f"Predictions:\n{predictions}\n"
    )

    # Сохранение результатов
    save_results(output_file, output_content)
    print(output_content)

if __name__ == "__main__":
    main()