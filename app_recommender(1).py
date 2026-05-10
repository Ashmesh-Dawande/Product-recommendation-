import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Product Recommender", page_icon="🛒", layout="wide")

# ── Load & build data ──────────────────────────────────────────────────────────

@st.cache_data
def load_all(path):
    df = pd.read_csv(path,compression='gzip')

    catalogue = (
        df.groupby(['product_id', 'product_category_name_english'])
        .agg(avg_price=('price', 'mean'),
             avg_rating=('review_score', 'mean'),
             total_orders=('order_id', 'count'))
        .reset_index()
    )

    profiles = (
        df.groupby('customer_unique_id')
        .agg(
            fav_cats=('product_category_name_english',
                      lambda x: x.value_counts().index[:3].tolist()),
            avg_spend=('price', 'mean'),
            state=('customer_state', 'first'),
            total_orders=('order_id', 'nunique')
        )
        .reset_index()
    )

    state_top = (
        df.groupby(['customer_state', 'product_category_name_english'])
        ['order_id'].count()
        .reset_index(name='cnt')
        .sort_values('cnt', ascending=False)
    )

    return df, catalogue, profiles, state_top


def get_recommendations(customer_id, catalogue, profiles, state_top, top_n, min_rating):
    row = profiles[profiles['customer_unique_id'] == customer_id]
    if row.empty:
        return None, None

    fav_cats   = row['fav_cats'].values[0]
    avg_spend  = row['avg_spend'].values[0]
    state      = row['state'].values[0]
    low        = avg_spend * 0.7
    high       = avg_spend * 1.3

    recs = catalogue[
        catalogue['product_category_name_english'].isin(fav_cats) &
        catalogue['avg_price'].between(low, high) &
        (catalogue['avg_rating'] >= min_rating)
    ].copy()

    # Fallback if not enough results
    if len(recs) < top_n:
        fallback_cats = state_top[state_top['customer_state'] == state]['product_category_name_english'].head(3).tolist()
        extra = catalogue[
            catalogue['product_category_name_english'].isin(fallback_cats) &
            (catalogue['avg_rating'] >= min_rating)
        ]
        recs = pd.concat([recs, extra]).drop_duplicates('product_id')

    if recs.empty:
        return {'avg_spend': avg_spend, 'state': state, 'fav_cats': fav_cats}, pd.DataFrame()

    recs['score'] = (
        0.6 * (recs['avg_rating'] / 5) +
        0.4 * (np.log1p(recs['total_orders']) / np.log1p(recs['total_orders'].max() + 1))
    )

    result = recs.sort_values('score', ascending=False).head(top_n).reset_index(drop=True)
    result.index += 1
    profile = {'avg_spend': avg_spend, 'state': state, 'fav_cats': fav_cats,
               'total_orders': int(row['total_orders'].values[0])}
    return profile, result[['product_id', 'product_category_name_english', 'avg_price', 'avg_rating', 'total_orders', 'score']]


# ── Sidebar ────────────────────────────────────────────────────────────────────

st.sidebar.title("🛒 Product Recommender")
st.sidebar.markdown("---")
csv_path   = st.sidebar.text_input("CSV file path", value="master_df.csv.gz")
top_n      = st.sidebar.slider("Number of recommendations", 3, 10, 5)
min_rating = st.sidebar.slider("Minimum rating", 1.0, 5.0, 4.0, 0.1)


# ── Load data ──────────────────────────────────────────────────────────────────

try:
    df, catalogue, profiles, state_top = load_all(csv_path)
except FileNotFoundError:
    st.error("❌ File not found. Make sure master_df.csv is in the same folder.")
    st.stop()


# ── Header ─────────────────────────────────────────────────────────────────────

st.title("🛒 Product Recommendation Engine")
st.caption("Rule-based engine · Olist Brazilian E-Commerce Dataset")
st.markdown("---")


# ── Platform stats ─────────────────────────────────────────────────────────────

st.subheader("Platform Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers",  f"{df['customer_unique_id'].nunique():,}")
c2.metric("Total Products",   f"{df['product_id'].nunique():,}")
c3.metric("Categories",       f"{df['product_category_name_english'].nunique()}")
c4.metric("Avg Review Score", f"{df['review_score'].mean():.2f} ⭐")
st.markdown("---")


# ── Customer selection ─────────────────────────────────────────────────────────

st.subheader("Find Recommendations")

top_customers = df['customer_unique_id'].value_counts().index[:40].tolist()
selected = st.selectbox("Select a customer", options=top_customers,
                        format_func=lambda x: x[:20] + "…")

manual_id = st.text_input("Or paste any Customer ID here", placeholder="Full customer ID…")
customer_id = manual_id.strip() if manual_id.strip() else selected


# ── Run recommendation ─────────────────────────────────────────────────────────

if st.button("Get Recommendations", type="primary"):
    profile, recs = get_recommendations(customer_id, catalogue, profiles, state_top, top_n, min_rating)

    if profile is None:
        st.warning("Customer not found in dataset.")
    elif recs.empty:
        st.warning("No matching products found. Try lowering the minimum rating.")
    else:
        # Customer profile
        st.markdown("### Customer Profile")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("State",        profile['state'])
        p2.metric("Avg Spend",    f"R${profile['avg_spend']:.2f}")
        p3.metric("Total Orders", profile['total_orders'])
        p4.metric("Price Band",   f"R${profile['avg_spend']*0.7:.0f} – R${profile['avg_spend']*1.3:.0f}")

        st.write("**Favourite Categories:**", ", ".join(profile['fav_cats']))
        st.markdown("---")

        # Recommendations table
        st.markdown("### Top Recommendations")
        display = recs.copy()
        display['product_category_name_english'] = display['product_category_name_english'].str.replace('_', ' ').str.title()
        display['avg_price']  = display['avg_price'].round(2)
        display['avg_rating'] = display['avg_rating'].round(2)
        display['score']      = display['score'].round(3)
        display.columns       = ['Product ID', 'Category', 'Avg Price (R$)', 'Avg Rating', 'Total Orders', 'Score']
        st.dataframe(display, use_container_width=True)

        # Bar chart
        fig = px.bar(recs, x='score', y='product_category_name_english',
                     orientation='h', color='score',
                     color_continuous_scale='Blues',
                     labels={'score': 'Score', 'product_category_name_english': 'Category'},
                     title='Recommendation Scores')
        fig.update_layout(yaxis={'autorange': 'reversed'}, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Overall charts ─────────────────────────────────────────────────────────────

st.subheader("Platform Analytics")
col1, col2 = st.columns(2)

with col1:
    top_cats = df['product_category_name_english'].value_counts().head(10).reset_index()
    top_cats.columns = ['Category', 'Orders']
    top_cats['Category'] = top_cats['Category'].str.replace('_', ' ').str.title()
    fig1 = px.bar(top_cats, x='Orders', y='Category', orientation='h',
                  title='Top 10 Categories by Orders', color='Orders',
                  color_continuous_scale='Blues')
    fig1.update_layout(yaxis={'autorange': 'reversed'}, coloraxis_showscale=False)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    state_orders = df.groupby('customer_state')['order_id'].count().nlargest(10).reset_index()
    state_orders.columns = ['State', 'Orders']
    fig2 = px.bar(state_orders, x='State', y='Orders',
                  title='Top 10 States by Orders', color='Orders',
                  color_continuous_scale='Blues')
    fig2.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)
