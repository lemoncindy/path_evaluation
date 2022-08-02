
Evaluation of paths' information diffusion capacity in online social networks

path_evaluation_information_diffusion_path is to rank paths in online social networks according to their role in information diffusion.

1 Introdution.
The workflow consists of 3 stages:
1.1 Network construction：with interactive records of users in a subnet in an online social network, allocate the contents of interactive records with topic vectors through LDA topic model.  Calculate interests of uses’ interaction, weigh the edges of the following relationship network to build the network representing users’ topic preferences via the calculation of multi-topic relationship strength (tRS), 
1.2 Calculating information diffusion capacity: calculate Information Diffusion Capacity of paths  based on relationship strength, interactive activity, users’ influence and users’ topic preferences. In the meantime, rank paths with two baseline methods including the method of edge centrality [1] and the connectivity-based method [2].
1.3 Methods comparing： the three methods are compared through plotting distribution of results and simulating information diffusion process with Susceptible-Infected–Recovered (SIR) model.


2 Requirements
package 
sklearn
networkx
matplotlib


3 Executation
To execute the program, you need to prepare a set of users' interactive records where a record including a user, a parent user from whom the user repost the post, the content of the posts and time or just use the dataset in directory 'data'.
And then you can run the file setup.py.

