
Evaluation of paths' information diffusion capacity in online social networks

path_evaluation_information_diffusion_path is to rank paths in online social networks according to their role in information diffusion.

1 Introdution.

The workflow consists of 3 stages:

1.1 Network construction：with interactive records of users in a subnet in an online social network, allocate the contents of interactive records with topic vectors through LDA topic model.  Calculate interests of uses’ interaction, weigh the edges of the following relationship network to build the network representing users’ topic preferences via the calculation of multi-topic relationship strength (tRS), 

1.2 Calculating information diffusion capacity: calculate Information Diffusion Capacity of paths  based on relationship strength, interactive activity, users’ influence and users’ topic preferences. In the meantime, rank paths with two baseline methods including the method of edge centrality [1] and the connectivity-based method [2].

1.3 Methods comparing： the three methods are compared through plotting distribution of results and simulating information diffusion process with Susceptible-Infected–Recovered (SIR) model.


2 Requirements 
scikit-learn >= 0.24.1
networkx >= 2.5
matplotlib >= 3.2.1


3 Executation

3.1 To execute the program, you need to prepare in a *.xlsx file which is a set of users' interactive records and a record including a user, a parent user from whom the user repost the post, the content of the posts and time or just use the dataset in directory 'data'.

3.2 In setup.py You need to change record_file_name 'tweet_records' into the name of your records, and the records_source 'Twitter' into you source_name, and you can run the file setup.py.


References:
[1] Brandes, U. (2001). A faster algorithm for betweenness centrality. The Journal of Mathematical Sociology, 25(2), 163–177. https://doi.org/10.1080/0022250X.2001.9990249
[2] Wu R, Zhou Y, Chen Z (2019) Identifying urban traffic bottlenecks with percolation theory. Urban Transport of China 17(01):96-101. https://doi.org/10.13813/j.cn11-5141/u.2019.0002
