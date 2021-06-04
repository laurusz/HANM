# Hierarchical Additive Noise Model for Many-to-One Causality Inference
## data available:
Synthetic_data.py: Generation of Synthetic Data.  
Real-world Data: Concrete Compressive Strength Data, Environmental Data and Auto MPG Data.

Results are shown in .\Result\HANM_result.xlsx.  

## code available:
HANM.py: Hierarchical Additive Noise Model (HANM) generalizes many-to-one causallity into an approximate pair relationship through the framework of a variational autoencoder to identify the causal relationship.  
PCA-ANM.py: PCA-ANM algorithms. Reduce the dimensions of multiple causes of many-to-one causality through Principal Component Analysis (PCA), and follow the second, third, and fourth phases of the HANM to identify the causal direction.  

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
    many-to-one causallity into an approximate pair relationship through 
    the framework of a variational autoencoder to identify the causal relationship.

    **Data Type**: Continuous

    Example:

        >>> data = create_data(n_causes=8, n_samples=500, sigma=0.0025, func='exp', seed=0)
        >>> result = HANM(data)   
        


### License
This project is licensed under the MIT License - see the LICENSE file for details.
