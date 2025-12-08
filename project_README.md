---
title: 'Assignment 6 Part 3: Multivariable Regression Project'
author: 'Mr. Berg, Lane Tech College Prep'
geometry: margin=1in
---

# Multivariable Linear Regression Project

**Due Date:** Presentations on **Tuesday, December 16, 2025** (during finals)

**Group Size:** Work alone or in groups of up to 4 students

---

## Project Overview

Now it's your turn! You'll find your own dataset, build a multivariable linear regression model, and present your findings to the class.

This project brings together everything you've learned:
- Finding and exploring real-world data
- Building multivariable linear regression models
- Evaluating model performance
- Communicating your findings

---

## Your Tasks

### 1. Find a Dataset (Due: 12/17/25)

Find a dataset with:
- **At least 50 rows** of data
- **At least 3 features** (X variables) that might predict something
- **One target variable** (Y) that you want to predict
- **All numeric data** (or data you can convert to numbers)

**Where to find datasets:**
- [Kaggle](https://www.kaggle.com/datasets) - huge collection, search for "regression"
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
- [Data.gov](https://data.gov/) - government datasets
- [Chicago Data Portal](https://data.cityofchicago.org/) - Data about Chicago
- [Google Dataset Search](https://datasetsearch.research.google.com/)

**Once you find a dataset, get Mr. Berg's approval before proceeding!**

---

## Dataset Ideas by Interest

Not sure what to look for? Here are some ideas organized by topic:

### Sports
- **NBA Player Stats** → Predict points per game from assists, rebounds, minutes played
- **FIFA Soccer Players** → Predict player value from age, skills, position
- **Baseball Stats** → Predict batting average from at-bats, walks, stolen bases
- **Olympic Athletes** → Predict medal count from training hours, age, country

### Entertainment & Media
- **Movie Box Office** → Predict revenue from budget, runtime, number of screens
- **Spotify Song Popularity** → Predict streams from tempo, danceability, energy
- **YouTube Videos** → Predict views from likes, comments, video length
- **Video Game Sales** → Predict sales from review score, platform, genre

### Health & Fitness
- **Calories Burned** → Predict calories from duration, heart rate, activity type
- **Sleep Quality** → Predict sleep score from hours, exercise, caffeine intake
- **Body Measurements** → Predict BMI from age, height, activity level
- **Life Expectancy** → Predict lifespan from GDP, healthcare spending, education

### Real Estate & Finance
- **Apartment Rent Prices** → Predict rent from bedrooms, location, square feet
- **Stock Prices** → Predict closing price from opening, volume, market cap
- **Used Cars** → Predict price from mileage, age, brand, horsepower
- **Airbnb Listings** → Predict nightly price from location, amenities, reviews

### Environment & Science
- **Air Quality** → Predict pollution level from temperature, humidity, traffic
- **Energy Consumption** → Predict usage from temperature, occupancy, time of day
- **Forest Fires** → Predict burned area from temperature, humidity, wind
- **Weather Patterns** → Predict rainfall from temperature, pressure, humidity

### Education
- **Student Performance** → Predict test scores from study hours, attendance, sleep
- **College Admissions** → Predict acceptance from GPA, test scores, activities
- **Course Grades** → Predict final grade from homework, quizzes, attendance

### Food & Nutrition
- **Restaurant Ratings** → Predict rating from price, cuisine, location
- **Wine Quality** → Predict quality score from acidity, sugar, alcohol content
- **Fast Food Nutrition** → Predict calories from fat, protein, carbs

---

## What to Submit

### 1. Python Code (`project.py`)
Your code should include:
- Loading and exploring the data
- Visualizing relationships between features and target
- Training/testing split
- Training the model
- Evaluating performance (R², RMSE)
- Making predictions
- **Comments explaining what you're doing!**

### 2. Presentation (slides or Google Slides)

**Your presentation should be 5-7 minutes and include:**

1. **Introduction (1 min)**
   - What dataset did you choose?
   - What are you trying to predict?
   - Why is this interesting?

2. **Your Data (1-2 min)**
   - How many samples? How many features?
   - Show 1-2 visualizations of your data
   - Any interesting patterns you noticed?

3. **Your Model (2-3 min)**
   - Which features did you use?
   - What were your coefficients?
   - Which feature was most important? How do you know?
   - Show an example prediction

4. **Results (1-2 min)**
   - What was your R² score?
   - What was your RMSE?
   - How accurate is your model?
   - Show actual vs predicted comparison (table or graph)

5. **Reflection (1 min)**
   - What went well?
   - What was challenging?
   - If you had more time, what would you improve?
   - One interesting thing you learned

---

## Grading Rubric (20 points total)

### Code (8 points)
- [ ] Loads and explores data appropriately (2 pts)
- [ ] Creates meaningful visualizations (2 pts)
- [ ] Correctly implements multivariable regression (2 pts)
- [ ] Evaluates model and interprets results (2 pts)

### Presentation (8 points)
- [ ] Clear explanation of dataset and goal (2 pts)
- [ ] Shows understanding of model and coefficients (2 pts)
- [ ] Discusses results and accuracy (2 pts)
- [ ] Professional delivery and stays within time (2 pts)

### Teamwork (4 points)
- [ ] All group members contributed
- [ ] All group members participate in presentation
- [ ] Work is well-organized and collaborative

**Note:** Even if your model doesn't work perfectly, you can still get full credit! We're grading your effort, understanding, and presentation - not whether you got a perfect R² score.

---

## Timeline

| Date | Milestone |
|------|-----------|
| **12/08/25** | Dataset selected and approved by Mr. Berg |
| **12/10/25** | Data loaded, explored, visualized |
| **12/11/25** | Model trained and evaluated |
| **12/12/25** | Code complete, start presentation |
| **12/16/25** | **Final presentations during finals** |

---

## Tips for Success

### Finding a Good Dataset
- ✓ Start simple - don't pick something too complicated
- ✓ Make sure all features are numeric (or can be converted)
- ✓ Avoid datasets with lots of missing values
- ✓ Pick something you're interested in - it's more fun!

### Building Your Model
- ✓ Start with all features, then try removing less important ones
- ✓ Create visualizations to understand your data first
- ✓ Don't worry if your R² isn't super high - real data is messy!
- ✓ Test your model on data it hasn't seen (train/test split)

### Creating Your Presentation
- ✓ Use visuals (graphs, charts) not just text
- ✓ Practice your timing - 5-7 minutes goes fast!
- ✓ Explain things clearly - pretend you're teaching the class
- ✓ Be ready to answer questions about your choices

### Working in Groups
- ✓ Divide up the work - maybe one person finds data, others code, others creat presentation
- ✓ Commit to GitHub regularly so Mr. Berg can see everyone contributed
- ✓ Help each other understand the concepts!

---

## Questions to Get You Started

Once you have your dataset, think about:

1. **What am I predicting?** (What's my Y variable?)
2. **What features might help predict it?** (What are my X variables?)
3. **Do I expect positive or negative relationships?** (Will coefficients be + or -?)
4. **Which feature do I think will be most important?** (Make a prediction!)
5. **How accurate do I hope to be?** (What would be a "good" R² score for this data?)

Write down your predictions BEFORE you build the model, then compare!

---

## Example Project Structure

```
my-project/
├── project.py              # Your code
├── data.csv                # Your dataset
└── README.md               # Brief description of your project
```

---

**Remember:** The goal is to apply what you've learned and communicate your findings clearly. Have fun with it and pick something you're genuinely curious about!

**Good luck! 🚀**
