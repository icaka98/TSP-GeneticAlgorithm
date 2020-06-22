<h1 align='center'>
  Travelling Salesman Problem with Genetic Algorithm
  <a href="https://github.com/sindresorhus/awesome"><img src="https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg" alt="Markdownify" width='120'>
  </a>
</h1>



<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)



<!-- ABOUT THE PROJECT -->
## About The Project

Genetic algorithms use an evolutionary theory approach to computationally approximate optimal solutions for NP-hard problems. This work focuses on two such examples - the 0/1 Knapsack Problem and the Travelling Salesman Problem. There exist numerous representations, crossover operators, mutation strategies and selection procedures that can successfully tackle both problems. In this repository, we implement several techniques from literature but also propose custom ones based on observations. All the newly introduced approaches are described in the paper provided. The paper also reports and comments on the computational results of the custom experiments.

Genetic Algorithms provide a wide variety of parameters to adjust in order to successfully and quickly reach optimality. These GA specifications vary greatly depending on the
problem definition and size. This report focuses on the 0/1 Knapsack and the Travelling Salesman problems. Our research shows that specific GA parameters are of greater interest based on the problem formulation. The 0/1 Knapsack section puts accent on the mutation and crossover properties of the GA. We show that a mutation that removes two items
and adds a one other is much more suitable for our problem than removing only one item. Moreover, we proved that a single-point crossover performs worse than our "swap-one" crossover in a complex 0/1 Knapsack setting. The TSP section introduces a modified crossover operator and emphasises on the general problem representation. We illustrate that the proposed CPMX crossover operator outperforms the other two algorithms from the literature given the correct mutation rate. Future work suggestions from this report are to thoroughly examine the performance of the CPMX over different GA approaches and formulations.


## Built With
This section lists the major frameworks that the project was built with.
* [Python](https://www.python.org/)


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- CONTACT -->
## Contact
Hristo Minkov - minkov.h@gmail.com

Zhecho Mitev - zhecho15@yahoo.com

Codebase Link: [https://github.com/icaka98/TSP-GeneticAlgorithm](https://github.com/icaka98/TSP-GeneticAlgorithm)




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=flat-square
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=flat-square
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=flat-square
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=flat-square
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: git_images/present.png
