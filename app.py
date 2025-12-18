import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from scipy.spatial import ConvexHull
from streamlit_gsheets import GSheetsConnection

SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1ch2PxWYFILX_hR6qE4sLKNjSq_Fg9A_SzbFstcz_hm4/edit"
# Replace this with the specific tab name (category) you want to load
WORKSHEET_NAME = "Ethnic"

st.set_page_config(page_title="Genetics Visualization", layout="wide")

@st.cache_data(ttl=600)
def load_data(url, worksheet):
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        
        df = conn.read(spreadsheet=url, worksheet=worksheet)
        
        def clean_hex(x):
            if isinstance(x, str):
                x = x.strip()
                if not x.startswith('#'):
                    return '#' + x
            return '#888888'

        for col in ['SubGroup Hexcode', 'Ethnic Group Hexcode', 'Ethnicity Hexcode']:
            if col in df.columns:
                df[col] = df[col].apply(clean_hex)
        return df
        
    except Exception as e:
        st.error(f"Error loading Google Sheet: {e}")
        return None

df_all = load_data(SPREADSHEET_URL, WORKSHEET_NAME)

if df_all is None:
    st.error(f"Could not load data. Please check the URL and Worksheet name.")
    st.stop()

st.sidebar.header("Configuration")

dimension_mode = st.sidebar.radio("Dimensions", ["2D", "3D"], index=0)
n_components = 3 if dimension_mode == "3D" else 2

all_families = sorted(df_all['Family'].dropna().unique())
selected_families = st.sidebar.multiselect(
    "Select Families to Run:",
    options=all_families,
    default=all_families 
)

if not selected_families:
    st.warning("Please select at least one family from the sidebar.")
    st.stop()

df = df_all[df_all['Family'].isin(selected_families)].copy()
n_samples = len(df)

st.sidebar.markdown(f"**Active Samples:** {n_samples}")

st.sidebar.subheader("Parameters")
perp = st.sidebar.slider("t-SNE Perplexity", 2, 50, min(30, n_samples - 1) if n_samples > 1 else 1)
n_neighbors = st.sidebar.slider("UMAP Neighbors", 2, 50, min(15, n_samples - 1) if n_samples > 1 else 2)

feature_cols = [c for c in df.columns if c.startswith('d') and c[1:].isdigit()]
X = df[feature_cols].values

min_samples_needed = 4 if dimension_mode == "3D" else 3
if n_samples < min_samples_needed:
    st.error(f"Not enough samples selected (need at least {min_samples_needed}) to perform {dimension_mode} dimensionality reduction.")
    st.stop()

with st.spinner(f'Calculating {dimension_mode} Projections...'):
    safe_n_comps = min(n_components, n_samples)
    pca = PCA(n_components=safe_n_comps)
    coords_pca = pca.fit_transform(X)

    tsne = TSNE(n_components=n_components, perplexity=perp, random_state=42)
    coords_tsne = tsne.fit_transform(X)

    reducer = umap.UMAP(
        n_components=n_components, 
        n_neighbors=n_neighbors, 
        min_dist=0.1, 
        metric='cosine',
        random_state=42
    )
    coords_umap = reducer.fit_transform(X)

coord_map = {
    'PCA': coords_pca,
    't-SNE': coords_tsne,
    'UMAP': coords_umap
}

def get_palette(items):
    colors = px.colors.qualitative.Dark24
    palette = {}
    for i, item in enumerate(items):
        palette[item] = colors[i % len(colors)]
    return palette

family_palette = get_palette(df['Family'].unique())
group_palette = get_palette(df['Group'].unique())

def build_figure(coords, method_name, dims):
    fig = go.Figure()
    
    def add_hull_layer(layer_name, group_col, color_source, show_legend=True):
        legend_marker = dict(size=10, color='grey', symbol='square')
        fig.add_trace(go.Scatter(
            x=[None], y=[None], 
            mode='markers',
            marker=legend_marker,
            name=f"{layer_name}s",
            legendgroup=layer_name,
            showlegend=show_legend
        ))
        
        groups = df[group_col].unique()
        for grp in groups:
            subset_mask = df[group_col] == grp
            
            threshold = 4 if dims == 3 else 3
            if subset_mask.sum() < threshold: continue
            
            sub_coords = coords[subset_mask]
            
            try:
                hull = ConvexHull(sub_coords)
                
                # Determine Color
                if isinstance(color_source, str):
                    color = df.loc[subset_mask, color_source].iloc[0]
                else:
                    color = color_source.get(grp, '#888888')

                if dims == 2:
                    hull_points = sub_coords[hull.vertices]
                    hull_points = np.vstack((hull_points, hull_points[0])) 
                    
                    fig.add_trace(go.Scatter(
                        x=hull_points[:, 0], y=hull_points[:, 1],
                        mode='lines', fill='toself',
                        line=dict(color=color, width=1),
                        opacity=0.1,
                        name=grp,
                        legendgroup=layer_name,
                        showlegend=False,
                        hoverinfo='name'
                    ))
                
                elif dims == 3:
                    x, y, z = sub_coords.T
                    i, j, k = hull.simplices.T
                    
                    fig.add_trace(go.Mesh3d(
                        x=x, y=y, z=z,
                        i=i, j=j, k=k,
                        color=color,
                        opacity=0.1,
                        name=grp,
                        legendgroup=layer_name,
                        showlegend=False,
                        hoverinfo='name'
                    ))

            except Exception: 
                continue

    add_hull_layer("Family Hull", 'Family', family_palette)
    add_hull_layer("Group Hull", 'Group', group_palette)
    add_hull_layer("SubGroup Hull", 'SubGroup', 'SubGroup Hexcode')

    
    hover_text = df.apply(lambda r: (
        f"<b>{r['Ethnicity']}</b><br>"
        f"SubGroup: {r['SubGroup']}<br>"
        f"Group: {r['Group']}<br>"
        f"Family: {r['Family']}"
    ), axis=1)
    
    marker_style = dict(
        color=df['Ethnicity Hexcode'],
        size=6 if dims == 3 else 8,
        line=dict(width=1, color='DarkSlateGrey') if dims == 2 else dict(width=0)
    )

    if dims == 2:
        fig.add_trace(go.Scatter(
            x=coords[:, 0], y=coords[:, 1],
            mode='markers',
            marker=marker_style,
            text=hover_text,
            hoverinfo='text',
            name='Individuals'
        ))
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
    else:
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode='markers',
            marker=marker_style,
            text=hover_text,
            hoverinfo='text',
            name='Individuals'
        ))
        fig.update_layout(
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            )
        )

    fig.update_layout(
        title=f"{method_name} {dims}D Projection (Samples: {n_samples})",
        template="plotly_white",
        height=800,
        legend=dict(groupclick="togglegroup", itemclick="toggle"),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

st.title("Genetics Dimensionality Reduction Analysis")
st.markdown("Select families in the sidebar to recalculate the projection based **only** on those populations.")

tab1, tab2, tab3 = st.tabs(["PCA", "t-SNE", "UMAP"])

with tab1:
    st.plotly_chart(build_figure(coords_pca, "PCA", n_components), use_container_width=True)

with tab2:
    st.plotly_chart(build_figure(coords_tsne, "t-SNE", n_components), use_container_width=True)
    
with tab3:
    st.plotly_chart(build_figure(coords_umap, "UMAP", n_components), use_container_width=True)
