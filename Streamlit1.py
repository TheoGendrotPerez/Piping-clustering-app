# -*- coding: utf-8 -*-
"""
Created on Mon May 26 16:35:00 2025

@author: tgend
"""
import streamlit as st
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import umap
from scipy.spatial import ConvexHull
import io
import base64

# --- Constants ---
DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

st.set_page_config(page_title="Equipment Clustering", layout="wide")

# --- Initialize session states ---
if 'selected_file' not in st.session_state:
    st.session_state.selected_file = None
if 'add_equipment' not in st.session_state:
    st.session_state.add_equipment = False
if 'run_clustering' not in st.session_state:
    st.session_state.run_clustering = False
if 'new_data' not in st.session_state:
    st.session_state.new_data = None
if 'final_df' not in st.session_state:
    st.session_state.final_df = None
if 'tab_selection' not in st.session_state:
    st.session_state.tab_selection = "Existing Files"

# --- Functions ---
def list_excel_files():
    return [f for f in os.listdir(DATA_FOLDER) if f.endswith(".xlsx")]

def load_dataset(file_path):
    try:
        df = pd.read_excel(file_path)
        st.success(f"✅ File loaded: {file_path}")
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return pd.DataFrame()

def save_dataset(df, filename):
    try:
        df.to_excel(os.path.join(DATA_FOLDER, filename), index=False)
        st.success("✅ Data saved successfully.")
    except Exception as e:
        st.error(f"Save error: {e}")

def highlight_new_rows(row):
    if 'new_data' in st.session_state and st.session_state.new_data is not None:
        if row['TagNumber'] in st.session_state.new_data['TagNumber'].values:
            return ['background-color: #ffff99'] * len(row)  # light yellow
    return [''] * len(row)

def download_link(df, filename="clustering_results.xlsx"):
    towrite = io.BytesIO()
    df.to_excel(towrite, index=False)
    towrite.seek(0)
    b64 = base64.b64encode(towrite.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download Excel file</a>'
    return href

# --- Home Page ---
st.title("Piping Clustering")

st.header("1️⃣ Choose an action")

# Radio buttons to replace tabs + reset run_clustering on change
st.markdown("""
<style>
    div[data-baseweb="radio"] label > span {
        font-size: 40px !important;  /* text size */
        font-weight: bold;            /* optional: bold text */
    }
    div[data-baseweb="radio"] {
        font-size: 20px !important;
    }
</style>
""", unsafe_allow_html=True)

selection = st.radio("Select an option:", ["📂 Facility Files", "➕ Add a new Facility File"], index=0)

if selection != st.session_state.tab_selection:
    st.session_state.tab_selection = selection
    st.session_state.run_clustering = False
    st.session_state.add_equipment = False
    st.session_state.selected_file = None
    st.session_state.new_data = None

if st.session_state.tab_selection == "📂 Facility Files":
    st.subheader("Facility files already loaded")
    files = list_excel_files()
    if files:
        selected = st.selectbox("Available files:", files)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Run clustering"):
                st.session_state.selected_file = os.path.join(DATA_FOLDER, selected)
                st.session_state.add_equipment = False
                st.session_state.run_clustering = True
                st.session_state.new_data = None
        with col2:
            if st.button("Add equipment to this file"):
                st.session_state.selected_file = os.path.join(DATA_FOLDER, selected)
                st.session_state.add_equipment = True
                st.session_state.run_clustering = False
                st.session_state.new_data = None
    else:
        st.warning("No files found in the 'data/' folder.")
        
elif st.session_state.tab_selection == "➕ Add a new Facility File":
    st.subheader("Import new facility file")
    uploaded = st.file_uploader("Upload a new Excel file", type="xlsx")
    if uploaded:
        new_filename = uploaded.name
        save_path = os.path.join(DATA_FOLDER, new_filename)
        with open(save_path, "wb") as f:
            f.write(uploaded.read())
        st.success(f"File '{new_filename}' saved.")
        if st.button("Run clustering on this new file"):
            st.session_state.selected_file = save_path
            st.session_state.add_equipment = False
            st.session_state.run_clustering = True
            st.session_state.new_data = None

# --- Add equipment ---
process_units = [
    "Separator", "Metering / Flow", "Pump", "Drain / Closed Drain",
    "Tank / Storage", "Gas Treatment / Fuel Gas", "Heater / Heat Exchanger", "Piping / Line / Manifold",
    "Flare / Vent / Relief", "Compression / Cooling", "Other / Unclassified"
]

fluid_types = [
    "Gas", "Oil", "Air", "Liquit", "Water", "Gross", "Glycol", "Waste Water"
]

if st.session_state.selected_file:

    if st.session_state.add_equipment:
        st.header("2️⃣ Add new equipment to this dataset")

        num_equipments = st.number_input("Number of equipment to input", min_value=1, max_value=20, value=1)

        temp_inputs = []
        validation_errors = False

        for i in range(num_equipments):
            st.subheader(f"Equipment {i+1}")
            tag_number = st.text_input(f"TagNumber {i+1}", key=f"tag_{i}")
            process_unit = st.selectbox(f"ProcessUnit {i+1}", process_units, key=f"unit_{i}")
            fluid_type = st.selectbox(f"FluidType {i+1}", fluid_types, key=f"fluid_{i}")
            op_pressure = st.text_input(f"Operating Pressure (psig) {i+1} (if empty or non-numerical, use Design Pressure)", key=f"press_{i}")
            op_temperature = st.text_input(f"Operating Temperature (°F) {i+1} (if empty or non-numerical, use Design Temperature)", key=f"temp_{i}")
            uninspected = st.selectbox(f"Uninspected (Yes/No) {i+1}", ['No', 'Yes'], key=f"uninspected_{i}")

            try:
                op_pressure_val = float(str(op_pressure).replace(',', '.'))
                op_temp_val = float(str(op_temperature).replace(',', '.'))
                if not tag_number:
                    validation_errors = True
                else:
                    temp_inputs.append({
                        "TagNumber": tag_number,
                        "ProcessUnit": process_unit,
                        "FluidType": fluid_type,
                        "OperatingPressure": op_pressure_val,
                        "OperatingTemperature": op_temp_val,
                        "Uninspected": uninspected
                    })
            except:
                validation_errors = True

        def save_and_go_results():
            if validation_errors:
                st.error("Please correct errors before running clustering.")
            elif len(temp_inputs) == 0:
                st.error("Please enter at least one valid equipment.")
            else:
                st.session_state.new_data = pd.DataFrame(temp_inputs)
                st.session_state.add_equipment = False
                st.session_state.run_clustering = True

        st.button("Clustering", on_click=save_and_go_results)

    # --- Clustering ---
    if st.session_state.run_clustering:
        st.header("2️⃣ Clustering results")

        # Load base dataset
        df = load_dataset(st.session_state.selected_file)
        if df.empty:
            st.stop()

        # Add new equipment if any
        if st.session_state.new_data is not None:
            df = pd.concat([df, st.session_state.new_data], ignore_index=True)

        # Preprocessing
        num_features = ['OperatingPressure', 'OperatingTemperature']
        cat_features = ['ProcessUnit', 'FluidType']

        for col in num_features:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

        df['CategoricalGroup'] = df[cat_features].astype(str).agg('-'.join, axis=1)
        df['CategoricalGroup'] = df['CategoricalGroup'].astype('category').cat.codes

        all_results = []
        corrosion_loop_id = 0
        temp_threshold = 18
        press_threshold = 14.7

        with st.spinner("Running clustering..."):
            for group_id, group_df in df.groupby('CategoricalGroup'):
                group_df = group_df.sort_values(by=num_features).reset_index(drop=True)
                subgroups = []
                current_subgroup = [group_df.iloc[0]]

                for i in range(1, len(group_df)):
                    prev = current_subgroup[-1]
                    curr = group_df.iloc[i]
                    if abs(curr['OperatingTemperature'] - prev['OperatingTemperature']) <= temp_threshold and \
                       abs(curr['OperatingPressure'] - prev['OperatingPressure']) <= press_threshold:
                        current_subgroup.append(curr)
                    else:
                        subgroups.append(pd.DataFrame(current_subgroup))
                        current_subgroup = [curr]
                subgroups.append(pd.DataFrame(current_subgroup))

                for subgroup_df in subgroups:
                    n_samples = len(subgroup_df)
                    if n_samples < 2:
                        subgroup_df['CorrosionLoop'] = corrosion_loop_id
                        corrosion_loop_id += 1
                        all_results.append(subgroup_df)
                        continue

                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(subgroup_df[num_features])
                    best_k = 1
                    best_silhouette = -1
                    best_labels = None
                    max_k = min(10, n_samples - 1)

                    for k in range(2, max_k + 1):
                        try:
                            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                            labels = kmeans.fit_predict(X_scaled)
                            if len(np.unique(labels)) < 2:
                                continue
                            sil_score = silhouette_score(X_scaled, labels)
                            if sil_score > best_silhouette:
                                best_k = k
                                best_silhouette = sil_score
                                best_labels = labels
                        except:
                            continue

                    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                    final_labels = kmeans_final.fit_predict(X_scaled)
                    subgroup_df['CorrosionLoop'] = final_labels + corrosion_loop_id
                    corrosion_loop_id += best_k
                    all_results.append(subgroup_df)

        final_df = pd.concat(all_results, ignore_index=True)
        final_df_filtered = final_df[['TagNumber', 'Uninspected', 'ProcessUnit', 'FluidType', 'OperatingPressure', 'OperatingTemperature', 'CorrosionLoop']]
        final_df_filtered.loc[:, num_features] = final_df_filtered[num_features].astype(int)
        st.session_state.final_df = final_df.copy()  # save in session

        # ============ DISPLAY REQUESTED ============

        st.success("Clustering completed.")

        if st.session_state.new_data is not None:
            st.write("### Newly input equipment:")
            st.dataframe(st.session_state.new_data)

        st.write("--------------------------------------------------------------------------------------")
        st.write("### Clustering results")
        st.markdown("_New equipment rows are highlighted in yellow._")
        st.dataframe(final_df_filtered.style.apply(highlight_new_rows, axis=1))
        st.write("--------------------------------------------------------------------------------------")

        reducer = umap.UMAP(random_state=42)
        embedding = reducer.fit_transform(final_df[num_features])
        final_df['UMAP1'] = embedding[:, 0]
        final_df['UMAP2'] = embedding[:, 1]

        fig = go.Figure()
        n_colors = final_df['CorrosionLoop'].nunique()
        colorscale = px.colors.qualitative.Dark24 if n_colors <= 24 else px.colors.sample_colorscale("Viridis", [i/n_colors for i in range(n_colors)])

        for i, (cluster_id, cluster_df) in enumerate(final_df.groupby('CorrosionLoop')):
            color = colorscale[i % len(colorscale)]

            fig.add_trace(go.Scatter(
                x=cluster_df['UMAP1'],
                y=cluster_df['UMAP2'],
                mode='markers',
                marker=dict(color=color, size=8),
                name=f'Corrosion Loop {cluster_id}',
                legendgroup=f'cluster_{cluster_id}',
                customdata=cluster_df[['TagNumber', 'ProcessUnit', 'FluidType', 'OperatingPressure', 'OperatingTemperature']].values,
                hovertemplate='TagNumber: %{customdata[0]}<br>ProcessUnit: %{customdata[1]}<br>FluidType: %{customdata[2]}<br>Pressure: %{customdata[3]}<br>Temp: %{customdata[4]}<extra></extra>'
            ))

            if len(cluster_df) >= 3:
                points = cluster_df[['UMAP1', 'UMAP2']].values
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                hull_points = np.append(hull_points, hull_points[:1], axis=0)

                fig.add_trace(go.Scatter(
                    x=hull_points[:, 0],
                    y=hull_points[:, 1],
                    mode='lines',
                    line=dict(color=color, width=2),
                    fill='toself',
                    fillcolor=color.replace(')', ',0.1)').replace('rgb', 'rgba'),
                    name=f'Corrosion loop {cluster_id}',
                    showlegend=False
                ))

        fig.update_layout(
            title="2D UMAP Projection of Equipment with Clustering",
            xaxis_title="UMAP1",
            yaxis_title="UMAP2",
            legend_title="Corrosion Loop",
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

        # ==== BUTTONS AT BOTTOM ====
        col_back, col_save, col_update = st.columns([1, 1, 2])

        with col_back:
            if st.button("🔙 Back to home"):
                st.session_state.selected_file = None
                st.session_state.add_equipment = False
                st.session_state.run_clustering = False
                st.session_state.new_data = None
                st.session_state.tab_selection = "📂 Existing Files"

        with col_save:
            def to_excel_download(df):
                import io
                buffer = io.BytesIO()
                df.to_excel(buffer, index=False)
                buffer.seek(0)
                return buffer

            if st.session_state.final_df is not None:
                buffer = to_excel_download(final_df_filtered)
                st.download_button(
                    label="💾 Download clustering results",
                    data=buffer,
                    file_name="clustering_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        with col_update:
            if st.session_state.new_data is not None:
                if st.button("💾 Update original file"):
                    save_dataset(
                        final_df.drop(columns=['UMAP1', 'UMAP2', 'CorrosionLoop', 'CategoricalGroup']),
                        os.path.basename(st.session_state.selected_file)
                    )
