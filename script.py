import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

os.makedirs('output', exist_ok=True)


# 1. ЗАГРУЗКА И ЧИСТКА
df = pd.read_csv('dataset.csv', index_col=0)
df = df.drop_duplicates()
df = df.dropna(subset=['artists', 'album_name', 'track_name'])
df['explicit'] = df['explicit'].astype(int)
print(f"Размер датасета после чистки: {df.shape}")

# 2. EDA
print("\n=== Целевая переменная (popularity) ===")
print(df['popularity'].describe())
print(f"Треков с popularity=0: {(df['popularity'] == 0).sum()}")

num_cols = ['popularity','duration_ms','danceability','energy','key','loudness',
            'mode','speechiness','acousticness','instrumentalness','liveness',
            'valence','tempo','time_signature','explicit']
print("\n=== Корреляция с popularity ===")
print(df[num_cols].corr()['popularity'].sort_values(ascending=False))

# ── График 1: Распределение целевой переменной
fig = px.histogram(df, x='popularity', nbins=50,
    title='Распределение популярности<br><span style="font-size:18px;font-weight:normal">Источник: Spotify | Правосторонний скос, среднее ~33</span>')
fig.update_xaxes(title_text='Популярность')
fig.update_yaxes(title_text='Количество')
fig.write_image('output/01_target_dist.png')

# ── График 2
corr = df[num_cols].corr()
fig = px.imshow(corr, text_auto='.2f', aspect='auto',
    color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
    title='Тепловая карта корреляций признаков<br><span style="font-size:18px;font-weight:normal">Источник: Spotify | loudness — наибольшая положит. корреляция</span>')
fig.update_xaxes(title_text='Признак')
fig.update_yaxes(title_text='Признак')
fig.write_image('output/02_corr_heatmap.png')

# ── График 3
genre_pop = (df.groupby('track_genre')['popularity']
               .mean()
               .sort_values(ascending=False)
               .head(20)
               .reset_index())
fig = px.bar(genre_pop, x='track_genre', y='popularity',
    title='Топ-20 жанров по средней популярности<br><span style="font-size:18px;font-weight:normal">Источник: Spotify | pop и latin лидируют</span>')
fig.update_xaxes(title_text='Жанр', tickangle=-35)
fig.update_yaxes(title_text='Средняя популярность')
fig.write_image('output/03_genre_popularity.png')

# ── График 4
df['pop_bucket'] = pd.cut(df['popularity'], bins=[0,20,40,60,80,100],
                           labels=['0-20','21-40','41-60','61-80','81-100'])
feats       = ['danceability','energy','loudness','speechiness','acousticness','instrumentalness']
feats_ru    = ['Танцевальность','Энергичность','Громкость','Доля речи','Акустика','Инструментал']
colors      = px.colors.qualitative.Plotly
fig = make_subplots(rows=2, cols=3, subplot_titles=feats_ru)
for idx, feat in enumerate(feats):
    r, c = divmod(idx, 3)
    for i, bucket in enumerate(['0-20','21-40','41-60','61-80','81-100']):
        vals = df[df['pop_bucket'] == bucket][feat].dropna()
        fig.add_trace(go.Box(y=vals, name=bucket, showlegend=(idx == 0),
                             marker_color=colors[i], boxpoints=False), row=r+1, col=c+1)
fig.update_layout(
    title_text='Аудио-признаки по бакетам популярности<br><span style="font-size:18px;font-weight:normal">Громкость и танцевальность растут с популярностью</span>',
    boxmode='group')
fig.write_image('output/04_features_by_bucket.png')

# ── График 5 
expl_pop = df.groupby('explicit')['popularity'].mean().reset_index()
expl_pop['Тип'] = expl_pop['explicit'].map({0: 'Без ненорм. лексики', 1: 'С ненорм. лексикой'})
fig = px.bar(expl_pop, x='Тип', y='popularity', color='Тип',
    title='Explicit vs Non-Explicit: средняя популярность<br><span style="font-size:18px;font-weight:normal">Источник: Spotify | explicit-треки немного популярнее</span>')
fig.update_xaxes(title_text='Тип трека')
fig.update_yaxes(title_text='Средняя популярность')
fig.update_layout(showlegend=False)
fig.write_image('output/05_explicit.png')

# ── График 6
sample = df.sample(3000, random_state=42)
fig = make_subplots(rows=1, cols=2,
    subplot_titles=['Громкость vs Популярность', 'Танцевальность vs Популярность'])
fig.add_trace(go.Scatter(x=sample['loudness'], y=sample['popularity'],
    mode='markers', marker=dict(size=4, opacity=0.4), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=sample['danceability'], y=sample['popularity'],
    mode='markers', marker=dict(size=4, opacity=0.4), showlegend=False), row=1, col=2)
fig.update_xaxes(title_text='Громкость (дБ)', row=1, col=1)
fig.update_xaxes(title_text='Танцевальность', row=1, col=2)
fig.update_yaxes(title_text='Популярность', row=1, col=1)
fig.update_yaxes(title_text='Популярность', row=1, col=2)
fig.update_layout(
    title_text='Признаки vs Популярность (диаграмма рассеяния)<br><span style="font-size:18px;font-weight:normal">Слабая линейная зависимость, есть нелинейные паттерны</span>')
fig.write_image('output/06_scatter.png')

print("Графики EDA сохранены.")

# 3. FEATURE ENGINEERING
df2 = df.copy()

df2['duration_min']     = df2['duration_ms'] / 60000
df2['energy_dance']     = df2['energy'] * df2['danceability']
df2['acoustic_vs_elec'] = df2['acousticness'] - df2['energy']
df2['mood_index']       = df2['valence'] + df2['danceability'] - df2['speechiness']
df2['loudness_norm']    = (df2['loudness'] - df2['loudness'].min()) / \
                           (df2['loudness'].max() - df2['loudness'].min())
df2['is_instrumental']  = (df2['instrumentalness'] > 0.5).astype(int)
df2['is_live']          = (df2['liveness'] > 0.8).astype(int)
df2['genre_pop_mean']   = df2.groupby('track_genre')['popularity'].transform('mean')
df2['genre_pop_std']    = df2.groupby('track_genre')['popularity'].transform('std')
df2['multi_artist']     = df2['artists'].str.contains(';').astype(int)

new_feats = ['duration_min','energy_dance','acoustic_vs_elec','mood_index',
             'loudness_norm','is_instrumental','is_live','genre_pop_mean',
             'genre_pop_std','multi_artist']

corr_new = (df2[new_feats + ['popularity']]
            .corr()['popularity']
            .drop('popularity')
            .sort_values(ascending=False))
print("\n=== Корреляция новых признаков с popularity ===")
print(corr_new)

# ── График 7
corr_df = corr_new.reset_index()
corr_df.columns = ['Признак', 'Корреляция']

rename_map = {
    'duration_min':     'Длит. (мин)',
    'energy_dance':     'Энергия×Танец',
    'acoustic_vs_elec': 'Акустика−Энергия',
    'mood_index':       'Индекс настроения',
    'loudness_norm':    'Громкость (норм.)',
    'is_instrumental':  'Инструментал',
    'is_live':          'Живое выступление',
    'genre_pop_mean':   'Попул. жанра (ср.)',
    'genre_pop_std':    'Попул. жанра (σ)',
    'multi_artist':     'Коллаборация',
}
corr_df['Признак'] = corr_df['Признак'].map(rename_map)
corr_df = corr_df.sort_values('Корреляция')

# Цвет по знаку: положительный — зелёный, отрицательный — оранжевый
corr_df['Цвет'] = corr_df['Корреляция'].apply(lambda x: 'Положительная' if x >= 0 else 'Отрицательная')

fig = px.bar(
    corr_df,
    x='Корреляция',
    y='Признак',
    orientation='h',
    color='Цвет',
    color_discrete_map={
        'Положительная': '#2ECC71',   # зелёный
        'Отрицательная': '#E74C3C',   # красный
    },
    title='Новые признаки vs Популярность<br><span style="font-size:18px;font-weight:normal">genre_pop_mean r=0.50 — сильнейший предиктор</span>'
)
fig.update_xaxes(title_text='Корреляция Пирсона (r)')
fig.update_yaxes(title_text='Признак')
fig.update_traces(texttemplate='')   # убираем цифры
fig.update_layout(legend_title_text='Тип корреляции')
fig.write_image('output/07_new_feat_corr.png')

# 4. ПОДГОТОВКА ДАННЫХ ДЛЯ ML
base_feats = ['duration_ms','explicit','danceability','energy','key','loudness',
              'mode','speechiness','acousticness','instrumentalness','liveness',
              'valence','tempo','time_signature']
all_feats = base_feats + new_feats

X = df2[all_feats].copy()
y = df2['popularity'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nОбучающая выборка: {X_train.shape} | Тестовая: {X_test.shape}")

# 5. БАЗОВАЯ МОДЕЛЬ
baseline_pred = np.full(len(y_test), y_train.mean())
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
baseline_r2   = r2_score(y_test, baseline_pred)
print(f"\n[Базовая (среднее)] RMSE={baseline_rmse:.3f}, R²={baseline_r2:.4f}")

# 6. ЭКСПЕРИМЕНТЫ С МОДЕЛЯМИ
def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    mae  = mean_absolute_error(y_te, pred)
    r2   = r2_score(y_te, pred)
    print(f"[{name}] RMSE={rmse:.3f} | MAE={mae:.3f} | R²={r2:.4f}")
    return {'Модель': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'model_obj': model}

results = []
results.append(evaluate("Ridge",
    Ridge(alpha=1.0),
    X_train_sc, y_train, X_test_sc, y_test))

results.append(evaluate("DecisionTree",
    DecisionTreeRegressor(max_depth=8, random_state=42),
    X_train, y_train, X_test, y_test))

results.append(evaluate("RandomForest",
    RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42),
    X_train, y_train, X_test, y_test))

results.append(evaluate("GradBoost",
    GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
    X_train, y_train, X_test, y_test))

results.append(evaluate("XGBoost",
    xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                     subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42, verbosity=0),
    X_train, y_train, X_test, y_test))

results.append(evaluate("LightGBM",
    lgb.LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                      subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42, verbose=-1),
    X_train, y_train, X_test, y_test))

results.append(evaluate("MLP",
    MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation='relu',
                 max_iter=300, early_stopping=True, random_state=42),
    X_train_sc, y_train, X_test_sc, y_test))

# 7. FEATURE IMPORTANCES
best_model = [r['model_obj'] for r in results if r['Модель'] == 'LightGBM'][0]

importances = pd.Series(best_model.feature_importances_, index=all_feats).sort_values(ascending=False)
print("\n=== Важность признаков (LightGBM) ===")
print(importances.head(10))

imp_df = importances.head(15).reset_index()
imp_df.columns = ['Признак', 'Важность']
imp_df = imp_df.sort_values('Важность')

fig = px.bar(imp_df, x='Важность', y='Признак', orientation='h',
    title='Топ-15 важных признаков (LightGBM)<br><span style="font-size:18px;font-weight:normal">genre_pop_mean — ключевой признак модели</span>')
fig.update_xaxes(title_text='Важность')
fig.update_yaxes(title_text='Признак')
fig.write_image('output/08_feature_importance.png')

# 8. СРАВНЕНИЕ МОДЕЛЕЙ
res_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'model_obj'} for r in results])
res_df = res_df.sort_values('RMSE')

fig = px.bar(res_df, x='Модель', y='RMSE', color='Модель',
    title='Сравнение моделей по RMSE<br><span style="font-size:18px;font-weight:normal">Чем меньше — тем лучше</span>',
    text=res_df['RMSE'].round(2))
fig.update_xaxes(title_text='Модель')
fig.update_yaxes(title_text='RMSE')
fig.update_traces(textposition='outside')
fig.write_image('output/09_model_comparison.png')

fig2 = px.bar(res_df, x='Модель', y='R2', color='Модель',
    title='Сравнение моделей по R²<br><span style="font-size:18px;font-weight:normal">Чем больше — тем лучше</span>',
    text=res_df['R2'].round(3))
fig2.update_xaxes(title_text='Модель')
fig2.update_yaxes(title_text='R²')
fig2.update_traces(textposition='outside')
fig2.write_image('output/10_model_r2.png')

# 9. КРОСС-ВАЛИДАЦИЯ ЛУЧШЕЙ МОДЕЛИ 

best_name = res_df.iloc[0]['Модель']
print(f"\n=== Кросс-валидация: {best_name} ===")
best_obj = [r['model_obj'] for r in results if r['Модель'] == best_name][0]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse = cross_val_score(best_obj, X, y, cv=kf,
                           scoring='neg_root_mean_squared_error', n_jobs=-1)
cv_r2   = cross_val_score(best_obj, X, y, cv=kf,
                           scoring='r2', n_jobs=-1)

print(f"RMSE по фолдам: {(-cv_rmse).round(3)}")
print(f"CV RMSE: {(-cv_rmse).mean():.3f} ± {(-cv_rmse).std():.3f}")
print(f"CV R²:   {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

cv_df = pd.DataFrame({
    'Фолд': [f'Фолд {i+1}' for i in range(5)],
    'RMSE': -cv_rmse,
    'R2':    cv_r2
})

fig = make_subplots(rows=1, cols=2, subplot_titles=['RMSE по фолдам', 'R² по фолдам'])
fig.add_trace(go.Bar(x=cv_df['Фолд'], y=cv_df['RMSE'],
    marker_color='#636EFA', showlegend=False), row=1, col=1)
fig.add_trace(go.Bar(x=cv_df['Фолд'], y=cv_df['R2'],
    marker_color='#EF553B', showlegend=False), row=1, col=2)
fig.add_hline(y=(-cv_rmse).mean(), line_dash='dash', line_color='white', row=1, col=1)
fig.add_hline(y=cv_r2.mean(),      line_dash='dash', line_color='white', row=1, col=2)
fig.update_layout(title_text=f'5-фолдовая кросс-валидация: {best_name}<br><span style="font-size:18px;font-weight:normal">Пунктир — среднее значение</span>')
fig.write_image('output/11_cv_results.png')

print("\n✅ Готово! Все графики сохранены в папку output/")