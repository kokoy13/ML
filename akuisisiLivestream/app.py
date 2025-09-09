from googleapiclient.discovery import build
import time
import csv

API_KEY = ''
VIDEO_ID = ''
youtube = build('youtube', 'v3', developerKey=API_KEY) 

# Step 1: Get Live Chat ID
def get_live_chat_id(video_id):
    response = youtube.videos().list(
        part='liveStreamingDetails',
        id=video_id
    ).execute()
    live_details = response['items'][0]['liveStreamingDetails']
    return live_details['activeLiveChatId']

# Step 2: Fetch Chat Messages
def get_chat_messages(live_chat_id):
    response = youtube.liveChatMessages().list(
        liveChatId=live_chat_id,
        part='snippet,authorDetails'
    ).execute()

    messages = []
    for item in response['items']:
        snippet = item['snippet']
        author = item['authorDetails']['displayName']

        # pastikan key 'displayMessage' ada
        if 'displayMessage' in snippet:
            msg = snippet['displayMessage']
            messages.append((author, msg))
        else:
            # untuk debug, biar tahu jenis pesan lain
            print("Pesan non-standar:", snippet.keys())

    return messages


# Step 3: Stream and Save to CSV
def stream_chat(video_id):
 live_chat_id = get_live_chat_id(video_id)
 with open('live_chat.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['User', 'Message'])
    print("Streaming started... Press Ctrl+C to stop.")
    try:
         while True:
            messages = get_chat_messages(live_chat_id)
            for user, msg in messages:
                print(f"{user}: {msg}")
                writer.writerow([user, msg])
            time.sleep(5)
    except KeyboardInterrupt:
        print("Streaming stopped.")
        
# Run
stream_chat(VIDEO_ID)