## How to run the HTML website locally:
1. First download the 'html' folder, and move or copy the 'apple_quality.csv' file into the html folder. <br>
Now, move into the html folder.
   ```(bash)
   cd html
3. Create and activate a new virtual environment:
 ```(bash)
  pip install virtualenv
  python3 -m venv
  source venv/bin/activate
 ```
3. Install the necessary libraries:
   ```(bash)
   pip install tensorflow flask joblib pandas seaborn scikit-learn scikeras
   ```
4. Run the backend:
   ```(bash)
   python3 app.py
5. The frontend is live at the following link: <br>
    <http://127.0.0.1:5500/templates/index.html>
6. Enter values for all fields and make a prediction!
