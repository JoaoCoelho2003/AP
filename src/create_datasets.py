import requests
from bs4 import BeautifulSoup
import random
import csv
import os
import signal
import sys
import time
import json
import google.generativeai as genai

def get_wikipedia_page(topic):
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&titles={topic}&format=json"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error: Unable to access page for {topic}")
        return None, None

    data = response.json()
    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), None)

    if not page or "extract" not in page:
        print(f"Error: No content found for {topic}")
        return None, None

    title = page.get("title", topic)
    content = page.get("extract", "")

    cleaned_text = " ".join(content.split())
    return title, cleaned_text

def scrape_wikipedia_page(topic):
    url = f"https://en.wikipedia.org/wiki/{topic}"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error: Unable to access page for {topic}")
        return None, None

    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.find("h1", {"id": "firstHeading"}).get_text()
    content = soup.find("div", {"class": "mw-parser-output"})

    if not content:
        return title, None

    paragraphs = content.find_all("p")
    text_content = " ".join(para.get_text() for para in paragraphs)
    cleaned_text = " ".join(text_content.split())
    return title, cleaned_text


def get_random_article_from_category(category_title):
    url = f"https://en.wikipedia.org/w/api.php?action=query&list=categorymembers&cmtitle=Category:{category_title}&cmlimit=500&format=json"
    response = requests.get(url)
    data = response.json()

    if "query" in data and "categorymembers" in data["query"]:
        members = data["query"]["categorymembers"]
        if members:
            article = random.choice(members)
            return article["title"]
    return None


def truncate_text(text, target_words=110):
    words = text.split()
    if len(words) <= target_words:
        return text

    truncated_words = words[:target_words]
    truncated_text = " ".join(truncated_words)

    last_dot = truncated_text.rfind(".")
    if last_dot != -1:
        truncated_text = truncated_text[: last_dot + 1]

    return truncated_text


def get_gemini_response(topic, model):
    try:
        prompt = f"""Please explain the topic: {topic} in a fluid, concise text about 100-120 words long.
		Make your response informative and educational without mentioning that you're an AI or that you're responding to a query."""

        response = model.generate_content(prompt)

        response_text = response.text.strip()

        if len(response_text.split()) > 120:
            response_text = truncate_text(response_text, 120)

        return response_text
    except Exception as e:
        print(f"Error getting Gemini response: {e}")
        return None


def save_to_csv(dataset, filename):
    try:
        with open(filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, delimiter="\t")
            writer.writerow(["Text", "Label"])
            writer.writerows(dataset)
        return True
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        return False


def generate_alternating_dataset(
    main_topics, filename="alternating_dataset.csv", api_key=None
):
    if api_key is None:
        print(
            "Error: Gemini API key is required for generating the alternating dataset."
        )
        return

    genai.configure(api_key=api_key)

    model_name = "gemini-1.5-flash"
    try:
        model = genai.GenerativeModel(model_name)
        print(f"Using Gemini model: {model_name}")
    except Exception as e:
        print(f"Error initializing {model_name}: {e}")
        print("Trying alternative model gemini-1.5-pro...")
        try:
            model = genai.GenerativeModel("gemini-1.5-pro")
            print("Using Gemini model: gemini-1.5-pro")
        except Exception as e2:
            print(f"Error initializing alternative model: {e2}")
            print("Trying final fallback model gemini-1.0-pro...")
            try:
                model = genai.GenerativeModel("gemini-1.0-pro")
                print("Using Gemini model: gemini-1.0-pro")
            except:
                print(
                    "Failed to initialize any Gemini model. Please check your API key and available models."
                )
                return

    combined_dataset = []
    seen_articles = set()

    if os.path.exists(filename):
        try:
            with open(filename, mode="r", newline="", encoding="utf-8") as file:
                reader = csv.reader(file, delimiter="\t")
                header = next(reader, None)
                combined_dataset = list(reader)

                for row in combined_dataset:
                    if len(row) >= 3:
                        seen_articles.add(row[0])

                print(f"Loaded existing dataset with {len(combined_dataset)} entries.")
        except Exception as e:
            print(f"Error loading existing dataset: {e}")
            print("Starting with an empty dataset.")

    consecutive_failures = 0
    max_consecutive_failures = 3

    try:
        while True:
            topic = random.choice(main_topics)
            article_title = get_random_article_from_category(topic)

            if article_title and article_title not in seen_articles:
                seen_articles.add(article_title)
                print(f"Processing article: {article_title} (from {topic})")

                title, wiki_text = scrape_wikipedia_page(article_title)

                if wiki_text:
                    truncated_wiki_text = truncate_text(wiki_text)

                    print(f"Getting Gemini response for: {title}")
                    ai_text = get_gemini_response(title, model)

                    if ai_text:
                        combined_dataset.append([truncated_wiki_text, "Human"])
                        combined_dataset.append([ai_text, "AI"])

                        consecutive_failures = 0

                        if save_to_csv(combined_dataset, filename):
                            print(
                                f"Saved {len(combined_dataset)} entries ({len(combined_dataset)//2} pairs)."
                            )
                        else:
                            print("Warning: Failed to save dataset.")
                    else:
                        print(f"Failed to get Gemini response for {title}")
                        seen_articles.remove(article_title)

                        consecutive_failures += 1
                        if consecutive_failures >= max_consecutive_failures:
                            print(
                                f"Stopping after {max_consecutive_failures} consecutive failures."
                            )
                            save_to_csv(combined_dataset, filename)
                            return
                else:
                    print(f"Failed to scrape {article_title}")
                    seen_articles.remove(article_title)

            time.sleep(2)

    except KeyboardInterrupt:
        print("\nProgram interrupted. Saving current progress...")
        save_to_csv(combined_dataset, filename)
        print("Dataset saved successfully. Exiting.")
        sys.exit(0)


def signal_handler(sig, frame):
    print("\nProgram interrupted. Saving current progress...")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    main_topics = [
        "Mathematics",
        "Biology",
        "Chemistry",
        "Physics",
        "Earth_science",
        "Ecology",
        "Microbiology",
        "Medicine",
        "Environmental_science",
        "Public_health",
    ]

    api_key = "YOUR_GEMINI_API_KEY"
    if not api_key:
        api_key = input("Enter your Google Gemini API key: ").strip()
        if not api_key:
            print("Error: Google Gemini API key is required for this script.")
            sys.exit(1)

    generate_alternating_dataset(
        main_topics, api_key=api_key, filename="./datasets/custom_dataset.csv"
    )
