import os
import random
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer

def gerar_nuvem_palavras(arquivo_mascara="cloud_mask.png",
                         dicionario_tokens_e_frequencia={'exemplo':1,
                                     'exemplo bigramas':2,
                                     'exemplo de trigrama':1,
                                     'ALTERAR':3}):
    """
    Gera uma nuvem de palavras com base em seu dicionário de palavras ou ngramas
    como a chave e a frequência de aparição do token como valor (inteiro).

    Parâmetros:
        dicionario_tokens_e_frequencia (dict): O dicionário de tokens e suas 
                                               respectivas frequências de 
                                               aparição nos textos.
        arquivo_mascara (str): O nome do arquivo da imagem de máscara. Pde ser:
                                            cloud_mask.png
                                            mapa_brasil_mask.png
                                            thumbs_up_mask.png        
                                            thumbs_down_mask.png
                                            <Outro arquivo de sua escolha>

    Exemplos de Uso:
        1. Para gerar uma nuvem de palavras na máscara mapa do brasil:
            gerar_nuvem_palavras(dicionario_tokens_e_frequencia=word_dict,
                                 arquivo_mascara='mapa_brasil_mask.png')
    """

    # Seletor de função de cor
    def color_function(mask):
        if mask == "mapa_brasil_mask.png":
            def color_func(word, font_size, position, orientation, random_state=None,**kwargs):
                return "hsl(190, 40%%, %d%%)" % random.randint(30, 60) #sky
            color_cont = (219, 236, 240)
        elif mask == "thumbs_down_mask.png":
            def color_func(word, font_size, position, orientation, random_state=None,**kwargs):
                return "hsl(0, 80%%, %d%%)" % random.randint(30, 60) #fake
            color_cont = (250, 209, 209)
        elif mask == "thumbs_up_mask.png":
            def color_func(word, font_size, position, orientation, random_state=None,**kwargs):
                return "hsl(130, 40%%, %d%%)" % random.randint(30, 60) #real
            color_cont = (219, 240, 223)
        else:
            def color_func(word, font_size, position, orientation, random_state=None,**kwargs):
                return "hsl(0, 0%%, %d%%)" % random.randint(60, 100) #grey
            color_cont = (219, 236, 240)
        return color_func, color_cont

    color_function, color_cont = color_function(arquivo_mascara)

    imagem_mascara = np.array(Image.open(os.path.join("data", "img", arquivo_mascara)))

    print(f"   * Um total de {len(dicionario_tokens_e_frequencia)} tokens foram "+
        "computadas a partir do conjunto de dados.\n")
    nuvem_palavras = WordCloud(
                            width=1080,
                            height=1080,
                            max_font_size=110,
                            background_color="white",
                            collocations=False,
                            mask=imagem_mascara,
                            contour_width=3,
                            contour_color = color_cont
                            ).generate_from_frequencies(dicionario_tokens_e_frequencia)
    nuvem_palavras.recolor(color_func=color_function, random_state=3)

    plt.figure(figsize=(12, 10))
    plt.imshow(nuvem_palavras, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    return nuvem_palavras, len(dicionario_tokens_e_frequencia)

def main():
    # Exemplo com uma frase completa
    texto = "Exemplo: Gerar uma nuvem de palavras usando texto completo"
    # Vetorização e contagem de frequência simples de bigramas:
    vectorizer = CountVectorizer(ngram_range=(2, 2))
    bag_of_words = vectorizer.fit_transform([a+' '+b for a,b in (ngrams(texto.split(),2))])
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    words_dict = dict(words_freq)
    print(words_dict)

    gerar_nuvem_palavras(dicionario_tokens_e_frequencia=words_dict,
                         arquivo_mascara='thumbs_up_mask.png')
    
    # Exemplo com Lista de bigramas
    bigramas = ['Olá mundo', 'teste exemplo', 'bigramas aqui', 'palavras distintas']
    # Vetorização e contagem de frequência simples de bigramas:
    vectorizer = CountVectorizer(ngram_range=(2, 2))
    bag_of_words = vectorizer.fit_transform(bigramas)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    words_dict = dict(words_freq)
    print(words_dict)

    gerar_nuvem_palavras(dicionario_tokens_e_frequencia=words_dict,
                         arquivo_mascara='mapa_brasil_mask.png')

    # Exemplo com dicionário de tokens unigramas e multigramas com frequências
    words_dict = {'Olá mundo':1,
                  'ALTERAR':4,
                  'bigramas aqui':2,
                  'palavras distintas de exemplo':2}
    gerar_nuvem_palavras(dicionario_tokens_e_frequencia=words_dict)

# Executar a função main
if __name__ == '__main__':
    from nltk import ngrams
    main()