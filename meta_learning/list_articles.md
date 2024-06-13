# Offline/Online Reinforcement Learning

[Excellent article](https://arxiv.org/pdf/2201.05433) to sum up Offline algorithms
I am lazy prefer to see [videos](https://slideslive.com/38935785/offline-reinforcement-learning-from-algorithms-to-practical-challenges) 
[offline_tuto](https://arxiv.org/pdf/2005.01643)

Offline reinforcement learning (RL) is a paradigm that learns from a fixed dataset of previously collected interactions, without requiring additional online data collection. This approach is particularly appealing for real-world applications where interacting with the environment is costly, dangerous, or infeasible. In this answer, we will provide an overview of offline learning algorithms in RL.

Key Challenges in Offline RL

Offline RL algorithms face several challenges:

Distributional Shift: The offline dataset may not reflect the distribution of the environment, leading to poor generalization.
Out-of-Distribution (OOD) Actions: The policy may choose actions that are not present in the training dataset, making it difficult to estimate the Q-function.
Limited Data: The offline dataset may be limited, making it challenging to learn a robust policy.
Popular Offline RL Algorithms

Several offline RL algorithms have been proposed to address these challenges:

Behavior Cloning: This algorithm learns a policy by imitating the behavior of an expert policy, without requiring additional online data collection.
Q-learning: This algorithm learns a Q-function by minimizing the Bellman error, using the offline dataset.
Actor-Critic Methods: These algorithms learn a policy and a value function simultaneously, using the offline dataset.
Generative Models: These algorithms use generative models to augment the offline dataset, making it more representative of the environment.
Recent Advances in Offline RL

Recent advances in offline RL have led to substantial improvements in the capabilities of offline RL algorithms. Some notable examples include:

Behavior Cloning with Regularization: This approach adds regularization terms to the policy loss to encourage exploration and reduce overfitting.
Offline Q-learning with Importance Sampling: This approach uses importance sampling to address the distributional shift problem and improve the estimation of the Q-function.
Offline Actor-Critic Methods with Normalization: This approach normalizes the offline dataset to reduce the impact of distributional shift and improve the learning process.
Conclusion

Offline RL algorithms have the potential to revolutionize the field of RL by enabling the learning of robust policies from fixed datasets. While there are still challenges to be addressed, recent advances in offline RL have shown promising results. As the field continues to evolve, we can expect to see more sophisticated algorithms and applications of offline RL in various domains.

## [DEMONSTRATION-REGULARIZED RL](https://arxiv.org/pdf/2310.17303)

## [MOPO: Model-based offline Policy Optimization](https://proceedings.neurips.cc/paper/2020/file/a322852ce0df73e204b7e67cbbef0d0a-Paper.pdf)

## [COMBO: Conservative Offline Model-Based Policy Optimization](https://proceedings.neurips.cc/paper/2021/file/f29a179746902e331572c483c45e5086-Paper.pdf)

## [Robust Reinforcement Learning using Offline Data](https://proceedings.neurips.cc/paper_files/paper/2022/file/d01bda31bbcd780774ff15b534e03c40-Paper-Conference.pdf)

## [A Closer Look at Offline RL Agents](https://proceedings.neurips.cc/paper_files/paper/2022/file/3908cadfcc99db12001eafb1207353e9-Paper-Conference.pdf)

## [Active Offline Policy Selection](https://proceedings.neurips.cc/paper_files/paper/2021/file/cec2346566ba8ecd04bfd992fd193fb3-Paper.pdf)

## [Bellman-consistent Pessimism for Offline Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2021/file/34f98c7c5d7063181da890ea8d25265a-Paper.pdf)


## [Survery Offline RL](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10078377)

## [IQL](https://arxiv.org/pdf/2110.06169)

## [CQL](https://arxiv.org/pdf/2006.04779)

https://proceedings.neurips.cc/paper_files/paper/2022/hash/ba1c5356d9164bb64c446a4b690226b0-Abstract-Conference.html
https://proceedings.neurips.cc/paper_files/paper/2023/hash/62d2cec62b7fd46dd35fa8f2d4aeb52d-Abstract-Datasets_and_Benchmarks.html
https://proceedings.neurips.cc/paper_files/paper/2022/hash/0b5669c3b07bb8429af19a7919376ff5-Abstract-Conference.html
https://proceedings.neurips.cc/paper/2021/hash/713fd63d76c8a57b16fc433fb4ae718a-Abstract.html
https://www.jair.org/index.php/jair/article/view/14174
https://proceedings.neurips.cc/paper_files/paper/2023/hash/c44a04289beaf0a7d968a94066a1d696-Abstract-Conference.html
https://arxiv.org/abs/2110.06169
https://arxiv.org/abs/2003.09398
https://arxiv.org/abs/2301.02328v2
https://paperswithcode.com/task/offline-rl
https://github.com/yihaosun1124/OfflineRL-Kit


https://paperswithcode.com/task/offline-rl
https://github.com/yihaosun1124/OfflineRL-Kit
https://ieeexplore.ieee.org/abstract/document/10078377
