import numpy as np
import uvicorn
from fastapi import FastAPI


app = FastAPI()

# dictionary of offer_id: its count as value
offers_count = {}
# dictionary of click_id: its corresponding offer_id
click_offer_dict = {}
# dictionary of offer_id: number of its conversions
offers_conversions = {}
# dictionary of offer_id: sum of its rewards
offers_rewards = {}


def upper_confidence_bound(offer_id):
    """Returns rpc (revenue per click) upper confidence bound for offer_id"""
    step = len(click_offer_dict)
    rpc = stats(offer_id)['rpc']

    if offer_id in offers_count:
        ucb = rpc + np.sqrt(1.5 * np.log(step) / offers_count[offer_id])
    else:
        ucb = 1e300

    return ucb


@app.on_event("startup")
def startup_event():
    offers_count.clear()
    click_offer_dict.clear()
    offers_conversions.clear()
    offers_rewards.clear()


@app.get("/sample/")
def sample(click_id: int, offer_ids: str) -> dict:
    """UCB sampling"""
    # Parse offer IDs
    offers_ids = [int(offer) for offer in offer_ids.split(",")]

    # Upper confidence bound sample offer ID
    offer_id = max(offers_ids, key=upper_confidence_bound)

    offers_count[offer_id] = offers_count.get(offer_id, 0) + 1
    click_offer_dict[click_id] = offer_id

    # Prepare response
    response = {
        "click_id": click_id,
        "offer_id": offer_id
    }

    return response


@app.put("/feedback/")
def feedback(click_id: int, reward: float) -> dict:
    """Get feedback for particular click"""
    # Response body consists of click ID
    # and accepted click status (True/False)

    is_conversion = bool(reward)
    offer_id = click_offer_dict[click_id]
    offers_conversions[offer_id] = offers_conversions.get(offer_id, 0) + is_conversion
    offers_rewards[offer_id] = offers_rewards.get(offer_id, 0) + reward

    response = {
        "click_id": click_id,
        "offer_id": offer_id,
        "is_conversion": is_conversion,
        "reward": reward
    }
    return response


@app.get("/offer_ids/{offer_id}/stats/")
def stats(offer_id: int) -> dict:
    """Return offer's statistics"""
    response = {
        "offer_id": offer_id,
        "clicks": offers_count.get(offer_id, 0),
        "conversions": offers_conversions.setdefault(offer_id, 0),
        "reward": offers_rewards.setdefault(offer_id, 0),
        "cr": offers_conversions[offer_id] / offers_count.get(offer_id, 1),
        "rpc": offers_rewards[offer_id] / offers_count.get(offer_id, 1)
    }
    return response


def main() -> None:
    """Run application"""
    uvicorn.run("upper_confidence_bound:app", host="localhost")


if __name__ == "__main__":
    main()
