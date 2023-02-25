import random
from typing import Callable, Dict, List, Tuple, TypeVar
import collections
import math
from typing import Any, DefaultDict, List, Set, Tuple

from util import *

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]


############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extrae características de palabras para una cadena x. Las palabras
    son delimitadas por espacios en blanco unicamente.
    @param string x:
    @return dict: representación en vector de características de x.
    Ejemplo: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # Inicio de tu código
    dic= collections.defaultdict(int)

    tx = x.split()
    for w in tx:
        dic[w]+=1

    return dic
    # Fin de tu código


############################################################
# Problem 3b: stochastic gradient descent

T = TypeVar("T")


def learnPredictor(
    trainExamples: List[Tuple[T, int]],
    validationExamples: List[Tuple[T, int]],
    featureExtractor: Callable[[T], FeatureVector],
    numEpochs: int,
    eta: float,
) -> WeightVector:
    """
    A partir de |trainExamples| y |validationExamples| (cada uno siendo una lista
    de parejas (x,y)), un |featureExtractor| para aplicar a x, y el número de
    épocas para entrenar |numEpochs|, el tamaño de paso |eta|, regresa el vector
    de pesos (vector de características disperas) aprendido.

    Debes implementar descenso de gradiente estocástico.

    Notas:
    - ¡Solo usa trainExamples para entrenar!
    - Debes llamar evaluatePredictor() sobre trainExamples y validationExamples para
    ver cómo te está llendo conforme aprendes después de cada época.
    - El predictor debe tener como salida +1 si el puntaje es precisamente 0.
    """
    weights = {}  # característica => peso
    increment(weights, eta, {})

    # Inicio de tu código

    def sign(x:float):
        return -1 if x < 0 else 1

    def predict(x : FeatureVector, w : WeightVector) -> float:
        return  dotProduct(w,x)       
    
    def loss(x : FeatureVector, w : WeightVector, y : int)-> float:
        return max(0, 1-predict(x, w)*y)

    def grad_loss(x : FeatureVector, w : WeightVector, y : int)->WeightVector:
        wg = {}
        if predict(x,w)*y <=0:
            increment(wg,y,x)
        return wg


    for epoch in range(numEpochs):

        # Iterar para cada ejemplo de entrenamiento
        for x, y in trainExamples:

            # Extraer las características del ejemplo
            features = featureExtractor(x)
            
            # Actualizar los pesos            
            increment(weights, eta, grad_loss(features, weights, y))

        # Evaluar
        trainError = evaluatePredictor(trainExamples, lambda x : predict(featureExtractor(x), weights))
        validationError = evaluatePredictor(validationExamples, lambda x : predict(featureExtractor(x), weights))
            
        print(f"Epoch {epoch}: train error = {trainError:.10f}, validation error = {validationError:.10f}")

        # Evaluar


    # Fin de tu código
    return weights


############################################################
# Problem 3c: generate test case


def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    """
    Regresa un conjunto de ejemplos (phi(x), y) aleatoriamente que sean clasificados
    correctamente por |weights|.
    """
    random.seed(42)

    # Regresa un solo ejemplo (phi(x), y).
    # phi(x) debe ser un diccionario cuyas llaves sean un subconjunto de las
    # llaves en weights y los valores pueden ser cualquier cosa con un puntaje
    # para el vector de pesos dado.
    # y debe ser 1 o -1 como lo clasifica el vector de pesos.
    # y debe ser 1 si el puntaje es precisamente 0.

    # Nota que el vector de pesos puede ser arbitrario durante las pruebas.
    def generateExample() -> Tuple[Dict[str, int], int]:
        # Inicio de tu código
        phi = {}
        sample = random.sample(list(weights), random.randint(0,len(weights)))
        for feature in sample:
            phi = {feature: random.random()} 
        y = 1 if dotProduct(weights, phi) >=0 else -1

        # Fin de tu código
        return phi, y

    return [generateExample() for _ in range(numExamples)]


############################################################
# Problem 3d: character features


def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    """
    Regresa una función que tome una cadena |x| y regrese un vector de
    características disperso que consiste de todos los n-gramas de |x| sin
    espacios asociados a los conteos de n-gramas.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    Puedes suponer que n >= 1.
    """

    def extract(x: str) -> Dict[str, int]:
        # Inicio de tu código
        features =  collections.defaultdict(int)
        x=x.replace(" ", "")
        for i in range(len(x) - n + 1):
            ngram = x[i:i+n]
            features[ngram]+= 1
        return features
        # Fin de tu código

    return extract

fe = extractCharacterFeatures(3)
sentence = "hello world"
fe(sentence)


############################################################
# Problem 3e:


def testValuesOfN(n: int):
    """
    Usa este código para probar distintos valores de n para extractCharacterFeatures
    Este código es exclusivo para las pruebas.
    Tu solución escrita completa debe estar en el archivo PDF.
    """
    trainExamples = readExamples("polarity.train")
    validationExamples = readExamples("polarity.dev")
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(
        trainExamples, validationExamples, featureExtractor, numEpochs=20, eta=0.01
    )
    outputWeights(weights, "weights")
    outputErrorAnalysis(
        validationExamples, featureExtractor, weights, "error-analysis"
 # type: ignore
    )
    trainError = evaluatePredictor(
        trainExamples,
 # type: ignore        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1),
    )
    validationError = evaluatePredictor(
        validationExamples,
 # type: ignore        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1),
    )
    print(
        (
            "Official: train error = %s, validation error = %s"
            % (trainError, validationError)
        )
    )


############################################################
# Problem 5: k-means
############################################################


def kmeans(
    examples: List[Dict[str, float]], K: int, maxEpochs: int
) -> Tuple[List, List, float]:
    """
    Realiza agrupamiento con K-medias sobre |examples|, donde cada ejemplo es un vector de
    características disperso.

    examples: lista de ejemplos, cada ejemplo es un diccionario string=>float representando un
              un vector disperso.
    K: número de grupos deseados. Supón que 0 < K <= |examples|.
    maxEpochs: máximo número de epocas para correr (debes terminar temprano solo si el algoritmo converge).
    Return: (lista de longitud K de centroides del agrupamiento,
             lista de asignaciones (es decir, si examples[i] pertenece a centers[j], entonces assignments[i] = j),
             pérdida final)
    """
    # Inicio de tu código
    raise Exception("Aún no implementada")
    # Fin de tu código
