from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import sklearn
import pickle


# Для WoE
def splitter(x, col_bondaries):
 
    for i in range(len(col_bondaries)):
        if (i > 0) and (x > col_bondaries[i-1]) and (x < col_bondaries[i]):
                return i
 
        if (i == 0) and (x <= col_bondaries[i]):
                return i
 
        if (i == len(col_bondaries) - 1) and (x > col_bondaries[i]):
                return i+1
        
def score_to_pd(score):
    '''
    Возвращает скор, преобразованный в pd
    '''
    pd_value = 1 / (1 + np.exp(-score))
    return pd_value

def logit_pd(pd_value):
    '''
    Возвращает колонку/лист/одномерный массив с логистически преобразованным PD
    '''
    logit = np.log(pd_value / (1 - pd_value))
    return logit


# Загрузка моделей и объекта масштабирования
# Сохраненная модель модуля Право.ru
with open('models/model_pr.pkl', 'rb') as f:
    model_pr = pickle.load(f)
# Коэффициенты модели модуля СПАРК-Интерфакс`
model_sp = {
    'intercept': -5.146401467044442,
    'coefs': [-0.63500042, -0.52792655, -0.69974848, -0.73613015, -0.53259876]
}
# Коэффициенты для интегральной модели
model_integr = {
    'intercept': 0.7060623578210927,
    'coefs': [0.19324893, 0.98244]
}

with open('models/standard_scaler_sp', 'rb') as f:
    scaler_sp = pickle.load(f)

with open('models/woe_dictionary_sp', 'rb') as f:
    woe_dict = pickle.load(f)

with open('models/woe_boundary_sp', 'rb') as f:
    bondaries = pickle.load(f)

# Список параметров
features_spark = scaler_sp.feature_names_in_.tolist()
# [
#     'p1500', 'p1250_std', 
#     'p2400_a', 'p1700_rt', 
#     'kz_days_rt',
# ]

# p1500 - чистая строка
# p1250_std = p1250 / p1500
# p2400_a = p2400 / p1600
# p1700_rt = p1700 / p1700_p
# kz_days_rt = (p2120 / p1520 * 365) / (p2120_p / p1520_p * 365)


features_pravo = model_pr.feature_names_
# [
#     'Ratio_ClaimSum_to_Assets', 'Ratio_ClaimSum_to_Revenue', 
#     'Ratio_ClaimSum_to_Capital', 'Ratio_ClaimSum_to_Funds',
#     'Ratio_ClaimSum_to_EBT',
# ]

# Ratio_ClaimSum_to_Assets = ClaimSum / p1600
# Ratio_ClaimSum_to_Revenue = ClaimSum / p2110
# Ratio_ClaimSum_to_Capital = ClaimSum / p1300
# Ratio_ClaimSum_to_Funds = ClaimSum / p1250
# Ratio_ClaimSum_to_EBT = ClaimSum / p2300

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route("/predict", methods = ["POST"])
def predict():
    print("Form data:", request.form)  # Логируем все данные формы
    input_vector_raw = [float(x) for x in request.form.values()]
    print("Raw input vector:", input_vector_raw)  # Логируем обработанный вектор

    indx = ['p1250', 'p1300', 'p1500', 'p1520', 'p1520_p', 'p1600', 'p1700', 'p1700_p', 'p2110', 'p2120', 'p2120_p', 'p2300', 'p2400', 'ClaimSum']

    if len(input_vector_raw) != len(indx):
        error_message = f"Expected {len(indx)} inputs, but got {len(input_vector_raw)}. Please check all input fields."
        return render_template("index.html", error_message=error_message)  # Возвращает страницу с сообщением об ошибке

    df_raw = pd.Series(input_vector_raw, index=indx)
    
    p1500 = df_raw['p1500']
    p1250_std = df_raw['p1250'] / df_raw['p1500']
    p2400_a = df_raw['p2400'] / df_raw['p1600']
    p1700_rt = df_raw['p1700'] / df_raw['p1700_p']
    kz_days_rt = (df_raw['p2120'] / df_raw['p1520'] * 365) / (df_raw['p2120_p'] / df_raw['p1520_p'] * 365)
    
    Ratio_ClaimSum_to_Assets = df_raw['ClaimSum'] / df_raw['p1600']
    Ratio_ClaimSum_to_Revenue = df_raw['ClaimSum'] / df_raw['p2110']
    Ratio_ClaimSum_to_Capital = df_raw['ClaimSum'] / df_raw['p1300']
    Ratio_ClaimSum_to_Funds = df_raw['ClaimSum'] / df_raw['p1250']
    Ratio_ClaimSum_to_EBT = df_raw['ClaimSum'] / df_raw['p2300']
    
    input_vector = [p1500
                    , p1250_std
                    , p2400_a
                    , p1700_rt
                    , kz_days_rt

                    , Ratio_ClaimSum_to_Assets
                    , Ratio_ClaimSum_to_Revenue
                    , Ratio_ClaimSum_to_Capital
                    , Ratio_ClaimSum_to_Funds
                    , Ratio_ClaimSum_to_EBT]
    
    
    df = pd.DataFrame(input_vector, index=features_spark+features_pravo).T
    
    

    # Логарифм
    df['p1500'] = np.log(np.log(df['p1500']))

    # Нормализация там, где необходимо
    df[features_spark] = scaler_sp.transform(df[features_spark])

    # WoE
    df_woe_vct = pd.DataFrame()
    for col in features_spark:
        df_woe_vct[col] = df[col].apply(lambda x: splitter(x, bondaries[col]))
        tmp_dict = woe_dict[col].set_index('group')['WoE'].to_dict()
        df_woe_vct[col] = df_woe_vct[col].map(tmp_dict)

    # Модельный скор, участвующий в интегральной модели. Вычисляется как линейная регрессия
    spark_score = model_sp['intercept']
    for i in range(0, len(features_spark)):
        spark_score = spark_score + model_sp['coefs'][i] * df_woe_vct.iloc[0, i]

    # Вероятность дефолта. Вычисляется из скора линейной регрессии, становясь лог регрессией
    spark_pd = score_to_pd(spark_score)
    print(spark_pd)

    pravoru_pd = model_pr.predict_proba(df[features_pravo])[:,-1].item()
    pravoru_score = logit_pd(pravoru_pd)

    # Интегральный скор
    integr_score = (model_integr['intercept'] + 
                    model_integr['coefs'][0] * spark_score + 
                    model_integr['coefs'][1] * pravoru_score)

    # Вероятность дефолта интегральной модели
    integr_pd = score_to_pd(integr_score)
    # Сравнение вероятности дефолта с порогом
    if integr_pd < 0.0187:
        loan_approval = "По результату рассмотренной заявки: кредитный займ одобрен"
    else:
        loan_approval = "По результату рассмотренной заявки: отказано в кредитном займе"

    return render_template(
        "index.html",
        prediction_text_1=f"Вероятность дефолта по данным экономических показателей составляет {np.round(spark_pd*100, 2)} %",
        prediction_text_2=f"Вероятность дефолта по данным судебных исков составляет {np.round(pravoru_pd*100, 2)} %",
        prediction_text=f"Вероятность дефолта обобщающей интегральной модели составляет {np.round(integr_pd*100, 2)} %",
        loan_approval_text=loan_approval  # Добавление сообщения о решении по кредиту
    )


if __name__ == "__main__":
    app.run('127.0.0.1', 5000, debug=True)