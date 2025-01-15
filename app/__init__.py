import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from googleapiclient.discovery import build
from groq import Groq
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
from transformers import pipeline
load_dotenv()
def create_app():
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes
    
    classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None, truncation=True)
    def get_comments(video_id, api_key):
        youtube = build('youtube', 'v3', developerKey=api_key)
        comments = []
        
        try:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText",
                maxResults=100
            ).execute()

            while response:
                for item in response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                    comments.append(comment)
                
                if 'nextPageToken' in response:
                    response = youtube.commentThreads().list(
                        part="snippet",
                        videoId=video_id,
                        pageToken=response['nextPageToken'],
                        maxResults=100
                    ).execute()
                else:
                    break
            
            return comments
        except Exception as e:
            print(f"Error fetching comments: {e}")
            return []

    def get_video_transcript(video_id):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['hi', 'en'])
            full_transcript = " ".join(entry['text'] for entry in transcript)
            return full_transcript
        except Exception as e:
            print(f"Failed to fetch transcription: {e}")
            return ""

    def batch_comments(comments, max_tokens=2048):
        batches = []
        current_batch = []
        current_length = 0

        for comment in comments:
            comment_length = len(comment.split())
            if current_length + comment_length > max_tokens:
                batches.append(current_batch)
                current_batch = [comment]
                current_length = comment_length
            else:
                current_batch.append(comment)
                current_length += comment_length

        if current_batch:
            batches.append(current_batch)

        return batches

    def chunk_text(text, max_tokens=6000):
        words = text.split()
        chunks = []
        current_chunk = []

        for word in words:
            if len(" ".join(current_chunk) + " " + word) > max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
            else:
                current_chunk.append(word)

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def analyze_video(video_id):
        # Initialize Groq client
        client = Groq(api_key=os.getenv('GROQ_API_KEY'))

        # Get transcript
        video_transcript = get_video_transcript(video_id)
        transcript_chunks = chunk_text(video_transcript, max_tokens=8000)

        # Summarize transcript in chunks
        transcript_summaries = []
        for chunk in transcript_chunks:
            summary_response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Provide a detailed summary of the given chunk of a YouTube video transcript."},
                    {"role": "user", "content": chunk}
                ],
                model="llama3-8b-8192",
            )
            transcript_summaries.append(summary_response.choices[0].message.content)
        
        # Combine transcript summaries
        combined_transcript_summary = " ".join(transcript_summaries)

        # Get comments and analyze
        comments = get_comments(video_id, os.getenv('GOOGLE_API_KEY'))
        comment_batches = batch_comments(comments)
        comment_summaries = []

        for batch in comment_batches:
            summary_response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "Summarize the following comments while keeping the detailed context."},
                    {"role": "user", "content": " ".join(batch)}
                ]
            )
            comment_summaries.append(summary_response.choices[0].message.content)

        # Combine comment summaries with transcript
        final_summary_response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": f"This is the summary of a YouTube video's transcript: {combined_transcript_summary}. A user has commented on the video. Your task is to analyze these comments in the context of the video transcript. Based on the comment content and its relation to the transcript, please provide detailed insights."},
                {"role": "user", "content": " ".join(comment_summaries)}
            ]
        )

        return final_summary_response.choices[0].message.content


    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/sentiment',methods=['GET'])
    def sentiment():
        video_id = request.args.get('videoId')
        
        try:
            comments = get_comments(video_id,os.getenv('GOOGLE_API_KEY'))
            import time

            sentences = ["I am not having a great day"]
            model_outputs = classifier(sentences)
            result = {model_outputs[0][i]['label']: [] for i in range(len(model_outputs[0]))}
            
            start_time = time.time()
            for i in comments:
                if time.time() - start_time > 240:
                    break
                model_outputs = classifier(i)
                result[model_outputs[0][0]["label"]].append(i)
            return jsonify({"result": result})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    @app.route('/analyze', methods=['GET'])
    def analyze():
        video_id = request.args.get('videoId')
        
        try:
            analysis_result = analyze_video(video_id)
            return jsonify({"result": analysis_result})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app

# This allows running the app directly for development
if __name__ == '__main__':
    app = create_app()
    app.run(debug=False)