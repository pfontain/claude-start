import anthropic
import json
from pathlib import Path
import logging
import argparse
from typing import Optional

# Global logger variable
logger: Optional[logging.Logger] = None

def setup_logger(enable_streaming_logging: bool) -> logging.Logger:
    """Configure and return a logger, or None if logging is disabled."""
    logger = logging.getLogger('poet_logger')
    logger.setLevel(logging.DEBUG)  # Set the overall logger level

    if enable_streaming_logging:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(stream_handler)

    file_handler = logging.FileHandler('app.log', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    return logger

# Path to the JSON file that stores the question-answer pairs
ANSWER_QUESTION_PAIR_DATA_JSON_PATH = Path("answer_question_pair_data.json")

def load_answer_question_pair_data() -> dict:
    """Load question-answer pairs from a JSON file if it exists."""
    global logger
    if ANSWER_QUESTION_PAIR_DATA_JSON_PATH.exists():
        try:
            with ANSWER_QUESTION_PAIR_DATA_JSON_PATH.open("r") as file:
                return json.load(file)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Error loading data from '{ANSWER_QUESTION_PAIR_DATA_JSON_PATH}': {e}")
            return {}
    return {}


def archive_answer_question_pair(question: str, answer: str, answer_question_pairs: dict) -> bool:
    """Archive a new question-answer pair into the dictionary."""
    answer_question_pairs[question] = answer
    return True


def save_answer_question_pair_data(answer_question_pairs: dict) -> bool:
    """Save the updated question-answer pairs to the JSON file."""
    global logger
    try:
        with ANSWER_QUESTION_PAIR_DATA_JSON_PATH.open("w") as file:
            json.dump(answer_question_pairs, file, indent=4)
        return True
    except OSError as e:
        logger.error(f"Error saving data in file '{ANSWER_QUESTION_PAIR_DATA_JSON_PATH}': {e}")
        return False


def ask_anthropic(question: str) -> tuple[bool, str | None]:
    """
    Ask a question to the Anthropics model and return a tuple (success_flag, response).
    
    Returns:
        - (True, answer) if the query was successful.
        - None if there was an issue with the query.
    """
    global logger
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            temperature=0,
            system="You are a world-class poet. Respond only with short poems.",
            messages=[
                {
                    "role": "user",
                    "content": question
                }
            ]
        )
        return True, response.content[0].text  # Return True and the answer
    except Exception as e:
        logger.error(f"Error querying Anthropics for question '{question}': {e}")
        return False, None


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog='claude poet',
        description='Answer questions as a short poem')
    parser.add_argument('--disable-streaming-logging', action='store_true', help="Disable streaming logging output.")
    args = parser.parse_args()
    return args

def main():
    global logger
    args = parse_arguments()
    logger = setup_logger(not args.disable_streaming_logging)
    logger.debug(f"Command line arguments: {args}")

    while True:
        question = input("Ask your question: ").strip()

        if question:
            break
        print("Hum ... did you mean to write something?")

    answer_question_pairs = load_answer_question_pair_data()

    # Check if the question is already archived
    if question in answer_question_pairs:
        print("I think I remember ...")
        print(answer_question_pairs[question])
    else:
        # Ask the question using Anthropics API
        success, response = ask_anthropic(question)

        if success:
            print(response)
            # Archive the answer and save the data
            if archive_answer_question_pair(question, response, answer_question_pairs):
                save_answer_question_pair_data(answer_question_pairs)
            else:
                logger.warning(f"Unable to archive answer to '{question}'")
        else:
            print('''I'm sorry, I can't answer your question right now''')


if __name__ == "__main__":
    main()