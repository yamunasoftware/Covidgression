# Covidgression

COVID-19 Death Toll Regression Model

## Information

This is a COVID-19 death toll regression model based on a single feature (Total People Tested for COVID). In the testing process, multiple features and combinations of features were tested in conjunction to determine the best combination of data features to predict the number of deaths caused. The application takes a ```.csv``` file of data, where a sample data file is provided under the name ```test_data.csv```. However, you can pass a filename of your liking to the application. The results of the model are outputted in the ```deaths.png``` file. The legend also shows the plotting of the data sample and the model predictions based on the length of time.

## Usage

To use the application, simple use the Bash or Batch files listed in the main directory. If you use Windows, run the batch script by running ```covidgression.bat <filename>``` in the command prompt. If you use MacOS, Linux, or GitBash, run the following command in the terminal: ```bash covidgression.sh <filename>```. Note the script takes a <i>N</i> number of days of COVID-19 data as a CSV file.

## Dependencies

- pandas
- matplotlib
- numpy

To install these dependencies simply run the pip commands listed below.

```
pip install pandas
pip install matplotlib
pip install numpy
```