import anthropic
import json
from pathlib import Path
import logging
import argparse
from typing import Optional

# Global logger variable
logger: Optional[logging.Logger] = None

# Path to the JSON file that stores the question-answer pairs
ANSWER_QUESTION_PAIR_DATA_JSON_PATH = Path("answer_question_pair_data.json")

# Path to the JSON file that stores cost data
COST_DATA_JSON_PATH = Path("cost_data.json")

# Claude 3.5 Sonnet price in dollars for a million input tokens
CLAUDE_3_5_SONNET_PRICE_DOLLARS_INPUT_PER_MILLION_OF_TOKENS = 3

# Claude 3.5 Sonnet price in dollars for a million output tokens
CLAUDE_3_5_SONNET_PRICE_DOLLARS_OUTPUT_PER_MILLION_OF_TOKENS = 15

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

def create_anthropic_request_arguments(question: str) -> dict:
    return {
        "model": "claude-3-5-sonnet-20240620",
        "system": "You are a world-class poet. Respond only with short poems.",
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    }

# TODO: Validate data
def load_cost_data() -> dict:
    """Load cost data from a JSON file if it exists."""
    global logger
    if COST_DATA_JSON_PATH.exists():
        try:
            with COST_DATA_JSON_PATH.open("r") as file:
                return json.load(file)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Error loading data from '{COST_DATA_JSON_PATH}': {e}")
            return {}
    return {}

def save_cost_data(cost_data: dict) -> bool:
    """Save the cost data to the JSON file."""
    global logger
    try:
        with COST_DATA_JSON_PATH.open("w") as file:
            json.dump(cost_data, file, indent=4)
        return True
    except OSError as e:
        logger.error(f"Error saving data in file '{COST_DATA_JSON_PATH}': {e}")
        return False

def get_anthropic_input_tokens_cost(input_tokens_count: int) -> float:
    return input_tokens_count * CLAUDE_3_5_SONNET_PRICE_DOLLARS_INPUT_PER_MILLION_OF_TOKENS / 1_000_000

def get_anthropic_output_tokens_cost(output_tokens_count: float) -> float:
    return output_tokens_count * CLAUDE_3_5_SONNET_PRICE_DOLLARS_OUTPUT_PER_MILLION_OF_TOKENS / 1_000_000

def log_anthropic_cost(question: str, output_tokens_average: float | None):
    global logger
    client = anthropic.Anthropic()

    request_arguments = create_anthropic_request_arguments(question)
    response = client.beta.messages.count_tokens(
        betas=["token-counting-2024-11-01"],
        model=request_arguments["model"],
        system=request_arguments["system"],
        messages=request_arguments["messages"]
    )

    logger.debug(f"input tokens count: {response.input_tokens}")

    input_cost = get_anthropic_input_tokens_cost(response.input_tokens)
    logger.debug(f"input cost: {input_cost}$")

    # Assumes the same number of tokens for output as input if there is not average
    # available.
    if output_tokens_average is None:
        output_tokens_average = response.input_tokens

    # Assumes that the number of tokens per response has a normal distribution
    estimated_output_cost = get_anthropic_output_tokens_cost(output_tokens_average)

    logger.debug(f"estimated output cost: {estimated_output_cost}$")

    estimated_total_cost = input_cost + estimated_output_cost
    logger.info(f"estimated total cost: {estimated_total_cost}$")

    return estimated_total_cost

def ask_anthropic(question: str) -> tuple[bool, str, int | None]:
    """
    Ask a question to the Anthropics model and return a tuple (success_flag, response, output_tokens_count).
    
    Returns:
        - (True, answer, output_tokens_count) if the query was successful.
        - None if there was an issue with the query.
    """
    global logger
    try:
        request_arguments = create_anthropic_request_arguments(question)
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=request_arguments["model"],
            max_tokens=1000,
            temperature=0,
            system=request_arguments["system"],
            messages=request_arguments["messages"]
        )
        return True, response.content[0].text, response.usage.output_tokens  # Return True, the answer, and the number of output tokens
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
        logger.info(f"question: {question}")
        cost_data = load_cost_data()
        log_anthropic_cost(question, cost_data.get("output_tokens_average", None))
        # Ask the question using Anthropics API
        success, response, output_tokens_count = ask_anthropic(question)

        if success:
            print(response)
            # Archive the answer and save the data
            if archive_answer_question_pair(question, response, answer_question_pairs):
                save_answer_question_pair_data(answer_question_pairs)
            else:
                logger.warning(f"Unable to archive answer to '{question}'")

            logger.debug(cost_data)
            if not cost_data:
                cost_data["total_output_tokens_count"] = output_tokens_count
                cost_data["output_tokens_sample_count"] = 1
            else:
                cost_data["total_output_tokens_count"] += output_tokens_count
                cost_data["output_tokens_sample_count"] += 1
            cost_data["output_tokens_average"] = cost_data["total_output_tokens_count"] / cost_data["output_tokens_sample_count"]

            save_cost_data(cost_data)
        else:
            print('''I'm sorry, I can't answer your question right now''')


if __name__ == "__main__":
    main()