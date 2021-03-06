# Project

- Designed and explained a Convolution Neural Network (CNN) in R keras
- Used graphics to visualize the concept of CNN
- The final report is available <a href="https://github.com/kh-w/mnist_fashion_m01/blob/main/report.pdf">here</a>

(Further model improvements)
- Ensemble CNNs with censorship to boost model accuracy

# Model improvement
<a href="https://github.com/kh-w/mnist_fashion_m01/blob/main/fashion_mnist_ensemble_benchmark.R">Ensemble</a>
1) Applied 4 censors seperately to the images
2) Trained a CNN for each censorship
3) Ensembled them and got 91.19% accuracy (0.14% improvement on the base model)

# Source of dataset
The fashion mnist dataset is obtainable at <a href="https://www.kaggle.com/zalando-research/fashionmnist">Zalando Research</a>. It is a dataset with 60000 (train) + 10000 (validate) = 70000 (total) Zalando's article images. 

# R version
R version 4.0.4

# Machine used
Macbook Air (13-inch, Early 2015)

# Replicate the results
<table>
  <tr>
    <td>Step 1:</td>
    <td>Install R and RStudio</td>
  </tr>
  <tr>
    <td>Step 2:</td>
    <td>Download R code file R_combine.R</td>
  </tr>
  <tr>
    <td>Step 3:</td>
    <td>Run it and install any necessary libraries</td>
  </tr>
  <tr>
    <td colspan="2"><i>(Note: Allow >12 hours runtime for any computers without Nvidia graphics card)</i></td>
  </tr>
  <tr>
    <td>Step 4:</td>
    <td>Download R markdown file Report.Rmd</td>
  </tr>
  <tr>
    <td colspan="2"><i>(Note: R code should be run to produce a RData file before producing the R markdown report)</td>
  </tr>
  <tr>
    <td>Step 5:</td>
    <td>Run it and install any necessary libraries</td>
  </tr>
</table>

# Lastly...
The project content is totally original and solely for academic purposes.<br>
This project is a great experience and fun to work on, so do not plagiarize.
