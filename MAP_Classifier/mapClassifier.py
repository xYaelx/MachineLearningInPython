import numpy as np
np.random.seed(42)

####################################################################################################
#                                            Part A
####################################################################################################

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset from which to compute the mean and mu (Numpy Array).
        - class_value : The class to calculate the mean and mu for.
        """
        self.data = dataset
        self.class_value = class_value
        self.class_data = dataset[dataset[:,-1]==class_value]

        self.mean = np.mean(self.class_data, axis = 0)
        self.std = np.std(self.class_data, axis=0)
        
    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """       
        return np.shape(self.class_data)[0]/np.shape(self.data)[0]
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood porbability of the instance under the class according to the dataset distribution.
        """
        px0 = normal_pdf(x[0], self.mean[0], self.std[0])
        px1 = normal_pdf(x[1], self.mean[1], self.std[1])

        return px0*px1
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_instance_likelihood(x) * self.get_prior()
    
class MultiNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset from which to compute the mean and mu (Numpy Array).
        - class_value : The class to calculate the mean and mu for.
        """
        self.data = dataset
        self.class_value = class_value
        self.class_data = dataset[dataset[:,-1]==class_value]
        self.class_data = self.class_data[:, :-1]
        self.mean = np.mean(self.class_data, axis = 0)
        self.cov = np.cov(self.class_data, rowvar = False) 
        
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return np.shape(self.class_data)[0]/np.shape(self.data)[0]
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        return multi_normal_pdf(x, self.mean, self.cov)

    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_instance_likelihood(x) * self.get_prior()
    
    

def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    deno = np.power(2*np.pi*std**2,0.5)
    e_power = -1*np.power(x-mean,2)/(2*std**2)
    return (1/deno)* np.exp(e_power)
    
    
def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variante normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    x = x[:-1]
    d = len(x)
    temp = np.power(2*np.pi,-0.5*d)
    temp *= np.linalg.det(cov)**(-0.5)
    e_power = -0.5*np.dot(np.dot((x-mean).T,np.linalg.inv(cov)),(x-mean))

    return np.dot(temp, np.exp(e_power))



####################################################################################################
#                                            Part B
####################################################################################################
EPSILLON = 1e-6 # == 0.000001 It could happen that a certain value will only occur in the test set.
                # In case such a thing occur the probability for that value will EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with la place smoothing.
        
        Input
        - dataset: The dataset from which to compute the probabilites (Numpy Array).
        - class_value : Compute the relevant parameters only for instances from the given class.
        """
        self.data = dataset
        self.class_value = class_value
        self.class_data = dataset[dataset[:,-1]==class_value]
    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return np.shape(self.class_data)[0]/np.shape(self.data)[0]

    def laplace_smoothing(self, value_x, attribute):
        data = self.class_data[:,attribute]
        nij = len(self.class_data[data == value_x])
        if nij == 0: #new attribute value that has not been seen
            return EPSILLON
        ni = len(data)
        vi = len(np.unique(data))
        return (nij+1)/(ni+vi)
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        res = 1
        for i in range(len(x)-1):
            res *= self.laplace_smoothing(x[i],i)
        
        return res
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_instance_likelihood(x) * self.get_prior()

    
####################################################################################################
#                                            General
####################################################################################################            
class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a postreiori classifier. 
        This class will hold 2 class distribution, one for class 0 and one for class 1, and will predicit and instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
        
        Input
            - An instance to predict.
            
        Output
            - 0 if the posterior probability of class 0 is higher 1 otherwise.
        """
        if self.ccd0.get_instance_posterior(x)>=self.ccd1.get_instance_posterior(x):
            return 0
        return 1
    
def compute_accuracy(testset, map_classifier):
    """
    Compute the accuracy of a given a testset and using a map classifier object.
    
    Input
        - testset: The test for which to compute the accuracy (Numpy array).
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / #testset size
    """
    true_rate = 0
    for instance in testset:
        if map_classifier.predict(instance) == instance[-1]:
            true_rate += 1

    return true_rate / len(testset)
    
            
            
        
