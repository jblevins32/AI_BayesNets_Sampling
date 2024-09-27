import sys

'''
WRITE YOUR CODE BELOW.
'''
from numpy import zeros, float32
#  pgmpy͏︅͏︀͏︋͏︋͏󠄌͏󠄎͏︀͏󠄋͏︊͏󠄏
import pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import random
#You are not allowed to use following set of modules from 'pgmpy' Library.͏︅͏︀͏︋͏︋͏󠄌͏󠄎͏︀͏󠄋͏︊͏󠄏
#
# pgmpy.sampling.*͏︅͏︀͏︋͏︋͏󠄌͏󠄎͏︀͏󠄋͏︊͏󠄏
# pgmpy.factors.*͏︅͏︀͏︋͏︋͏󠄌͏󠄎͏︀͏󠄋͏︊͏󠄏
# pgmpy.estimators.*͏︅͏︀͏︋͏︋͏󠄌͏󠄎͏︀͏󠄋͏︊͏󠄏

def make_security_system_net():
    """
        Create a Bayes Net representation of the above security system problem. 
        Use the following as the name attribute: "H","C", "M","B", "Q", 'K",
        "D"'. (for the tests to work.)
    """
    
    # Instantiate Bayesian network object
    BayesNet = BayesianNetwork()
    
    # Add nodes to the network
    BayesNet.add_node("H") # The event that Spectre hires professional hackers
    BayesNet.add_node("C") # The event that Spectre buys Contra
    BayesNet.add_node("M") # The event that Spectre hires mercenaries
    BayesNet.add_node("B") # The event that Bond is guarding M at the time of the kidnapping
    BayesNet.add_node("Q") # The event that Q’s database is hacked and the cipher is compromised
    BayesNet.add_node("K") # The event that M gets kidnapped and has to give away the key
    BayesNet.add_node("D") # The event that Spectre succeeds in obtaining the “Double-0” files
    
    # Add edges between the nodes (parent, child)
    BayesNet.add_edge("H","Q")
    BayesNet.add_edge("C","Q")
    BayesNet.add_edge("M","K")
    BayesNet.add_edge("B","K")
    BayesNet.add_edge("Q","D")
    BayesNet.add_edge("K","D")

    return BayesNet


def set_probability(bayes_net):
    """
        Set probability distribution for each node in the security system.
        Use the following as the name attribute: "H","C", "M","B", "Q", 'K",
        "D"'. (for the tests to work.)
    """

    # Given conditional probability distributions (false, true)
    cpd_h = TabularCPD('H', 2, values=[[0.5], [0.5]]) # Spectre will not be able to find and hire skilled professional hackers (call this false) with a probability of 0.5.
    cpd_c = TabularCPD('C', 2, values=[[0.7], [0.3]]) # Spectre will get their hands on Contra (call this true) with a probability of 0.3.
    cpd_m = TabularCPD('M', 2, values=[[0.2], [0.8]]) # Spectre will be unable to hire the mercenaries (call this false) with a probability of 0.2.
    cpd_b = TabularCPD('B', 2, values=[[0.5], [0.5]]) # Since Bond is also assigned to another mission, the probability that he will be protecting M at a given moment (call this true) is just 0.5!
    cpd_qhc = TabularCPD('Q', 2, values=[[0.95, 0.75, 0.45, 0.1], \
                    [0.05, 0.25, 0.55, 0.9]], evidence=['H', 'C'], evidence_card=[2, 2]) # The professional hackers will be able to crack Q’s personal database (call this true) without using Contra with a probability of 0.55. However, if they get their hands on Contra, they can crack Q’s personal database with a probability of 0.9. In case Spectre can not hire these professional hackers, their less experienced employees will launch a cyberattack on Q’s personal database. In this case, Q’s database will remain secure with a probability of 0.75 if Spectre has Contra and with a probability of 0.95 if Spectre does not have Contra.
    cpd_kmb = TabularCPD('K', 2, values=[[0.25, 0.99, 0.05, 0.85], \
                    [0.75, 0.01, 0.95, 0.15]], evidence=['M', 'B'], evidence_card=[2, 2]) # When Bond is protecting M, the probability that M stays safe (call this false) is 0.85 if mercenaries conduct the attack. Else, when mercenaries are not present, it the probability that M stays safe is as high as 0.99! However, if M is not accompanied by Bond, M gets kidnapped with a probability of 0.95 and 0.75 respectively, with and without the presence of mercenaries.
    cpd_dqk = TabularCPD('D', 2, values=[[0.98, 0.65, 0.4, 0.01], \
                    [0.02, 0.35, 0.6, 0.99]], evidence=['Q', 'K'], evidence_card=[2, 2]) # With both the cipher and the key, Spectre can access the “Double-0” files (call this true) with a probability of 0.99! If Spectre has none of these, then this probability drops down to 0.02! In case Spectre has just the cipher, the probability that the “Double-0” files remain uncompromised is 0.4. On the other hand, if Spectre has just the key, then this probability changes to 0.65.
  
    # Add the cpds to the network
    bayes_net.add_cpds(cpd_h, cpd_c, cpd_m, cpd_b, cpd_qhc, cpd_kmb, cpd_dqk)
    
    return bayes_net


def get_marginal_double0(bayes_net):
    """
        Calculate the marginal probability that Double-0 gets compromised.
    """

    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['D'], joint=False)
    double0_prob = marginal_prob['D'].values
    
    return double0_prob[1]


def get_conditional_double0_given_no_contra(bayes_net):
    """
        Calculate the conditional probability that Double-0 gets compromised
        given Contra is shut down.
    """

    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['D'],evidence={'C':1}, joint=False)
    double0_prob = conditional_prob['D'].values

    return double0_prob[1]


def get_conditional_double0_given_no_contra_and_bond_guarding(bayes_net):
    """
        Calculate the conditional probability that Double-0 gets compromised
        given Contra is shut down and Bond is reassigned to protect M.
    """

    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['D'],evidence={'C':1,'B':0}, joint=False)
    double0_prob = conditional_prob['D'].values
    
    return double0_prob[1]


def get_game_network():
    """
        Create a Bayes Net representation of the game problem.
        Name the nodes as "A","B","C","AvB","BvC" and "CvA".  
    """
    
    # Instantiate Bayesian network object
    BayesNet = BayesianNetwork()
    
    # Add nodes to the network
    BayesNet.add_node("A")
    BayesNet.add_node("B")
    BayesNet.add_node("C") 
    BayesNet.add_node("AvB") 
    BayesNet.add_node("BvC") 
    BayesNet.add_node("CvA")
    
    # Add edges between the nodes (parent, child)
    BayesNet.add_edge("A","AvB")
    BayesNet.add_edge("A","CvA")
    BayesNet.add_edge("B","AvB")
    BayesNet.add_edge("B","BvC")
    BayesNet.add_edge("C","BvC")
    BayesNet.add_edge("C","CvA")

    # Given conditional probability distributions
    skill_levels = [[0.15],[0.45],[0.30],[0.10]]
    skill_differences = [[0.1,0.2,0.15,0.05,0.6,0.1,0.2,0.15,0.75,0.6,0.1,0.2,0.9,0.75,0.6,0.1],
                         [0.1,0.6,0.75,0.9,0.2,0.1,0.6,0.75,0.15,0.2,0.1,0.6,0.05,0.15,0.2,0.1],
                         [0.8,0.2,0.1,0.05,0.2,0.8,0.2,0.1,0.1,0.2,0.8,0.2,0.05,0.1,0.2,0.8]]
    
    cpd_A = TabularCPD('A', 4, values=skill_levels)
    cpd_B = TabularCPD('B', 4, values=skill_levels)
    cpd_C = TabularCPD('C', 4, values=skill_levels)
    
    # Conditional probabilities given the players in the game's skills (priors/parents)
    cpd_AvB = TabularCPD('AvB', 3, values=skill_differences, evidence = ['A','B'], evidence_card = [4,4])
    cpd_BvC = TabularCPD('BvC', 3, values=skill_differences, evidence = ['B','C'], evidence_card = [4,4])
    cpd_CvA = TabularCPD('CvA', 3, values=skill_differences, evidence = ['C','A'], evidence_card = [4,4])
    
    BayesNet.add_cpds(cpd_A, cpd_B, cpd_C, cpd_AvB, cpd_BvC, cpd_CvA)

    # Print the probability tables
    # for cpd in BayesNet.get_cpds():
    #     print(f"CPD of {cpd.variable}:")
    #     print(cpd)
    
    return BayesNet


def calculate_posterior(bayes_net):
    """
        Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
        Return a list of probabilities corresponding to win, loss and tie likelihood.
    """
    posterior = [0,0,0]
    
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['BvC'],evidence={'AvB':0,'CvA':2}, joint=False)
    posterior = conditional_prob['BvC'].values
    print(posterior)
    return posterior # list 


def Gibbs_sampler(bayes_net, initial_state):
    """
        Complete a single iteration of the Gibbs sampling algorithm 
        given a Bayesian network and an initial state value. 
        
        initial_state is a list of length 6 where: 
        index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
        index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
        
        Returns the new state sampled from the probability distribution as a tuple of length 6.
        Return the sample as a tuple. 

        Note: You are allowed to calculate the probabilities for each potential variable
        to be sampled. See README for suggested structure of the sampling process.
    """
    sample = tuple(initial_state)    

    variable_index = random.randinit(0,5) # Your chosen variable
    print(variable_index)
    
    num_iters = 1 # One iteration for this problem
    for _ in range(num_iters):
        if variable_index == 0:
            # Sample A͏︅͏︀͏︋͏︋͏󠄌͏󠄎͏︀͏󠄋͏︊͏󠄏
            prob = P()
            sample[variable_index] = max(prob)
        elif variable_index == 1:
            # Sample B͏︅͏︀͏︋͏︋͏󠄌͏󠄎͏︀͏󠄋͏︊͏󠄏
        elif variable_index == 2:
            # Sample C͏︅͏︀͏︋͏︋͏󠄌͏󠄎͏︀͏󠄋͏︊͏󠄏
        elif variable_index == 3:
            # Sample AvB
        elif variable_index == 4:
            # Sample BvC͏︅͏︀͏︋͏︋͏󠄌͏󠄎͏︀͏󠄋͏︊͏󠄏
        elif variable_index == 5:
            # Sample CvA
        else:
            raise ValueError("Variable index out of range")

    return sample


def MH_sampler(bayes_net, initial_state):
    """
        Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
        initial_state is a list of length 6 where: 
        index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
        index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
        Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """
    A_cpd = bayes_net.get_cpds("A")      
    AvB_cpd = bayes_net.get_cpds("AvB")
    match_table = AvB_cpd.values
    team_table = A_cpd.values
    sample = tuple(initial_state)    
    # TODO: finish this function͏︅͏︀͏︋͏︋͏󠄌͏󠄎͏︀͏󠄋͏︊͏󠄏
    raise NotImplementedError    
    return sample


def compare_sampling(bayes_net, initial_state):
    """
        Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge.
    """    
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs 
    MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH
    # TODO: finish this function͏︅͏︀͏︋͏︋͏󠄌͏󠄎͏︀͏󠄋͏︊͏󠄏
    raise NotImplementedError        
    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count


def sampling_question():
    """
        Question about sampling performance.
    """
    # TODO: assign value to choice and factor͏︅͏︀͏︋͏︋͏󠄌͏󠄎͏︀͏󠄋͏︊͏󠄏
    raise NotImplementedError
    choice = 2
    options = ['Gibbs','Metropolis-Hastings']
    factor = 0
    return options[choice], factor


def return_your_name():
    """
        Return your name from this function
    """
    return "Jacob Blevins"

# Testing the functions
if __name__ == "__main__":
    BayesNet = get_game_network()
    
    # bayes_net = set_probability(BayesNet)
    # prob1 = get_marginal_double0(bayes_net)
    # prob2 = get_conditional_double0_given_no_contra(bayes_net)
    # prob3 = get_conditional_double0_given_no_contra_and_bond_guarding(bayes_net)
    
    # print(prob1,prob2,prob3)
