import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import spacy
nlp = spacy.load("en_core_web_sm")

import random
from PIL import Image
from bokeh.io import show, output_notebook
from bokeh.models import Plot, Range1d, MultiLine, Circle
from bokeh.models.graphs import from_networkx
from bokeh.plotting import figure
from bokeh.models import BoxZoomTool, ResetTool, PanTool
from bokeh.models import HoverTool
import requests
from bs4 import BeautifulSoup

page_selection = st.sidebar.radio(label="Label",options=["Recommender Engine","Bokeh"])



if page_selection == 'Recommender Engine':
    st.sidebar.subheader("Perfume Recommender Engine")



    st.title('Perfume Recommender Engine')

    user_input = st.text_input("Input the theme you want", 'Fresh')

    df = pd.read_csv('df_network_analysis.csv')
    df1 = df[['names' , 'value']]


    #Network Creation
    G = nx.Graph()
    G = nx.from_pandas_edgelist(df1, 'names' , 'value')

    node_color = []

    accord_list = ['aromatic', 'amber', 'woody', 'citrus', 'fresh spicy', 'leather',
        'vanilla', 'fruity', 'almond', 'sweet', 'warm spicy', 'powdery',
        'ozonic', 'musky', 'rose', 'oud', 'coconut', 'green', 'tuberose',
        'white floral', 'floral', 'tobacco', 'honey', 'whiskey', 'beeswax',
        'smoky', 'animalic', 'earthy', 'caramel', 'soapy', 'balsamic',
        'chocolate', 'cacao', 'coffee', 'lavender', 'lactonic', 'fresh',
        'soft spicy', 'conifer', 'iris', 'mineral', 'mossy',
        'cinnamon', 'cherry', 'patchouli', 'tropical', 'salty', 'marine',
        'yellow floral', 'herbal', 'Champagne', 'rum', 'nan']

    for node in G:
        if node in accord_list:
            node_color.append('red')
        else: 
            node_color.append('blue')  

    nx.draw(G, node_color=node_color, with_labels=True)
    plt.show()


    similarity_list = []
    for x in accord_list:
        similarity_list.append(nlp(x).similarity(nlp(user_input)))
        


    d = {'accord': accord_list, 'similarity': similarity_list}
    df = pd.DataFrame(data=d)
    df = df.sort_values('similarity', ascending=False)


    #st.markdown("We recommend to use " + list(G.neighbors(df['accord'].iloc[0]))[random.randint(0,len(list(G.neighbors(df['accord'].iloc[0]))) - 1)])
    recommended_perfume1 = list(G.neighbors(df['accord'].iloc[0]))[random.randint(0,len(list(G.neighbors(df['accord'].iloc[0]))) - 1)]
    recommended_perfume2 = list(G.neighbors(df['accord'].iloc[1]))[random.randint(0,len(list(G.neighbors(df['accord'].iloc[1]))) - 1)]
    recommended_perfume3 = list(G.neighbors(df['accord'].iloc[2]))[random.randint(0,len(list(G.neighbors(df['accord'].iloc[2]))) - 1)]
    recommended_perfume4 = list(G.neighbors(df['accord'].iloc[3]))[random.randint(0,len(list(G.neighbors(df['accord'].iloc[3]))) - 1)]
    recommended_perfume5 = list(G.neighbors(df['accord'].iloc[4]))[random.randint(0,len(list(G.neighbors(df['accord'].iloc[4]))) - 1)]
    
    similarity_list = []
    for x in accord_list:
        similarity_list.append(nlp(x).similarity(nlp(user_input)))

    d = {'accord': accord_list, 'similarity': similarity_list}
    df = pd.DataFrame(data=d)
    df = df.sort_values('similarity', ascending=False)

    recs = [recommended_perfume1,recommended_perfume2,recommended_perfume3,recommended_perfume4,recommended_perfume5]
    #Image Code
    images = []
    for word in recs:
        list=[]
        url = 'https://www.google.com/search?q={0}&tbm=isch'.format(word)
        content = requests.get(url).content
        soup = BeautifulSoup(content, 'html.parser')
        images1 = soup.findAll('img')

        for image in images1:
            list.append(image.get('src'))
            
        images.append(list[1])

 ############################################
    st.markdown("Recommended Perfumes:")

    for x in range(5):
        col1, mid, col2 = st.columns([3,1,20])
        with col1:
            st.image(images[x-1],width=100)
        with col2:
            st.header(recs[x-1])

    st.dataframe(df.sort_values('similarity', ascending=False).head(5))

    ProdExp = ''

    if df['similarity'].iloc[0] <= .4:
        ProdExp = 'Because the similarity is low, this theme has the possibility for product exploration'
    else:
        ProdExp = 'This theme has been covered by the recommendation'
    
    st.header(ProdExp)

elif page_selection == "Bokeh":
    st.sidebar.subheader("")
    st.title('Interactive Network')

    df = pd.read_csv('df_network_analysis.csv')
    df = df.dropna()
    accords_list = df['value'].to_list()
    df1 = df[['names' , 'value']]
    G = nx.Graph()
    G = nx.from_pandas_edgelist(df1, 'names' , 'value')

    node_color = []

    accord_list = ['aromatic', 'amber', 'woody', 'citrus', 'fresh spicy', 'leather',
        'vanilla', 'fruity', 'almond', 'sweet', 'warm spicy', 'powdery',
        'ozonic', 'musky', 'rose', 'oud', 'coconut', 'green', 'tuberose',
        'white floral', 'floral', 'tobacco', 'honey', 'whiskey', 'beeswax',
        'smoky', 'animalic', 'earthy', 'caramel', 'soapy', 'balsamic',
        'chocolate', 'cacao', 'coffee', 'lavender', 'lactonic', 'fresh',
        'soft spicy', 'conifer', 'iris', 'mineral', 'mossy',
        'cinnamon', 'cherry', 'patchouli', 'tropical', 'salty', 'marine',
        'yellow floral', 'herbal', 'Champagne', 'rum', 'nan']

    for node in G:
        if node in accord_list:
            node_color.append('red')
        else: 
            node_color.append('blue')  
        
    #node_color = [node_color[node] for node in sorted(node_color)]

    #node_color = [node_color[node] for node in node_color]

    nx.draw(G, node_color=node_color, with_labels=True)
    plt.show()



    nx.draw(G, node_color=node_color, with_labels=False)
    plt.show()
    #pos = nx.spring_layout(G)


    nx.draw(G, node_color=node_color, with_labels=True)
    plt.show()

    from matplotlib.pyplot import figure

    nx.draw_shell(G, node_color=node_color, with_labels=True)
    plt.show()

    perfumes_unique = [df[i].unique().tolist() for i in df.columns][1]
    accords_unique = [df[i].unique().tolist() for i in df.columns][2]

    accords_dictionary = dict.fromkeys(accords_unique, 'red')
    perfume_dictionary = dict.fromkeys(perfumes_unique, 'blue')

    accords_dictionary.update(perfume_dictionary)
    color_dictionary = accords_dictionary

    nx.set_node_attributes(G, color_dictionary, 'node_color')

    plot = Plot(plot_width=700, plot_height=500,
            x_range=Range1d(-1.1,1.1), y_range=Range1d(-1.1,1.1))

    plot.title.text = "Top 200 Perfumes"

    from bokeh.models import HoverTool
    plot.add_tools(BoxZoomTool(), ResetTool(), PanTool(),
                HoverTool(tooltips=[('Name', '@index')]))

    graph_renderer = from_networkx(G, nx.spring_layout, scale=1, center=(0,0))
    graph_renderer.node_renderer.glyph = Circle(size=10, fill_color='node_color')
    graph_renderer.edge_renderer.glyph = MultiLine(
                                                    line_alpha=0.4,
                                                    line_width=0.4)
    plot.renderers.append(graph_renderer)
    output_notebook()
    show(plot)

    st.bokeh_chart(plot, use_container_width=True)
