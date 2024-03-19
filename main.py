from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle

app = FastAPI()
tf_idf_similarity = pickle.load(open('tf_idf_cosine_similarity.pkl','rb'))

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", 
                   "https://shan7030.github.io/sta-frontend/",
                   "https://shan7030.github.io"],  # Add the URL of your frontend application
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

def fetch_all_course_data():
    all_courses_df = pd.read_csv('./all_courses_data.csv')
    concatenated_course_data_copy = all_courses_df.copy()
    concatenated_course_data_copy = concatenated_course_data_copy[concatenated_course_data_copy['course_title'].apply(lambda x: all(ord(char) < 128 for char in x))]

    return concatenated_course_data_copy

all_courses_data = fetch_all_course_data()

# Function to recommend courses based on cosine similarity
def recommend_courses(course_title, cosine_sim_matrix, df, top_n=5):
    # Find the index of the course title in the DataFrame
    idx = df.index[df['course_title'] == course_title].tolist()[0]
    
    # Get the cosine similarity scores for the given course
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    
    # Sort the courses based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the top n similar courses (excluding itself)
    top_similar_courses = sim_scores[1:top_n+1]
    
    # Get the indices of the top similar courses
    course_indices = [idx for idx, _ in top_similar_courses]
    # Get the course names corresponding to the indices
    recommended_courses = df.iloc[course_indices]
    recommended_courses.fillna(-1, inplace=True)

    return recommended_courses

# API for returning a list of results
@app.get("/results/")
async def get_results(course_title: str = Query(...)):
    """
    Function to return the recommendations when selected a course
    """
    # Here you can implement your logic to fetch results based on the category
    # For demonstration, I'll just return some dummy data
    cosine_vector = tf_idf_similarity

    recommended_courses_data = recommend_courses(course_title, cosine_vector, all_courses_data)
    result_df = recommended_courses_data[['course_title', 'course_id', 'course_url', 'course_instructor', 'course_rating', 'course_duration']]
    result_dict_list = result_df.to_dict('records')
    return result_dict_list

# API for returning a list of results
@app.get("/all-courses/")
async def get_all_courses():
    """
    Function to return all the courses
    """
    result_df = all_courses_data[['course_title', 'course_id']]
    result_dict_list = result_df.to_dict('records')
    return result_dict_list