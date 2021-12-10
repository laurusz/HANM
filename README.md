# Hierarchical Additive Noise Model for Many-to-One Causality Inference
## data available:
Synthetic_data.py: Generation of Synthetic Data.  
Real-world Data: Concrete Compressive Strength Data, Environmental Data and Auto MPG Data.

## code available:
HANM.py: Hierarchical Additive Noise Model (HANM) generalizes many-to-one causallity into an approximate pair relationship to identify the causal relationship.  

### dependencies 
- Keras                    2.4.3 
- tensorflow               2.2.0 
- numpy                    1.18.5  

The code is tested with python 3.7 on Windows 10. 
GPU: NVIDIA GeForce RTX 2080Ti 

HANM (data):

    :param data: Data of many-to-one causality {Xi,...,Xn,Y}.  
    :param seed: The random seed.  
    :return: The causality of Xi and Y. 0: Y->Xi; 1: Xi->Y; -1: Non-identifiable.

    

    HANM algorithm.

    **Description**: Hierarchical Additive Noise Model (HANM) generalizes 
    many-to-one causallity into an approximate pair relationship to identify 
    the causal relationship.

    **Data Type**: Continuous

    Example:

        >>> data = create_data(n_causes=8, n_samples=500, sigma=0.0025, func='exp', seed=0)
        >>> result = HANM(data)   
        


### License
This project is licensed under the MIT License - see the LICENSE file for details.
