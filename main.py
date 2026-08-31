"""
GUI for my mail spam filter
"""

# Importing important modules
import html
import joblib
import pandas as pd
import streamlit as st
from datetime import timedelta
from sklearn.model_selection import train_test_split

# naming the broweser tab
st.set_page_config(page_title="Mail - Spam Filter - IU Internationale Hochshule", layout="wide")

def route_email(p_spam: float):
    """
    Same threshold applied as planned previously:
     - If model is sure that message is spam less than 10%, message gets directly to Inbox
     - If model is sure that message is span more that 95%, message gets directly to Spam
     - If model is sure that message is spam less than 40%, message gets directly to Review folder
     and being added +1 hour to Review by deadline
     - If model is sure that message is spam less than 75%, message gets directly to Review folder
     and being added +7 hour to Review by deadline
     - otherwise, message gets to Review folder and being added +24 hours to Review by deadline
    """
    if p_spam < .1:
        return "Inbox", None
    if p_spam > .95:
        return "Spam", None
    if p_spam < .45:
        return "Review folder", 1
    if p_spam < .75:
        return "Review folder", 7
    return "Review folder", 24

# Loading the trained model
@st.cache_resource
def load_model():
    return joblib.load("models/model.joblib")


@st.cache_data
def load_mock_mailbox():
    model = load_model()
    # Loading the test data. As there is no API yet
    X_test = pd.read_csv("data/processed/X_test.csv")
    probs = model.predict_proba(X_test)[:, 1]

    # loading the raw messages as X_test has only numerical (vectorized) data
    df = pd.read_csv("data/raw/SMSSpamCollection.csv", sep="\t", header=None, names=["label", "message"])
    # preprocessing the data in the same way as during model training in order to get the same X_test messages
    df["message"] = df["message"].apply(html.unescape)
    df = df.drop_duplicates(subset=["message", "label"]).reset_index(drop=True)
    y_full = df["label"].map({"ham": 0, "spam": 1})
    _, X_test_raw, _, _ = train_test_split(df, y_full, train_size=0.80, random_state=42, stratify=y_full)
    messages = X_test_raw["message"].reset_index(drop=True)

    mailbox = pd.DataFrame({"message": messages, "p_spam": probs})

    """
    Stratified sample, so every folder actually has something in it for the demo.
    Otherwise, a plain random sample would leave the review folder almost empty,
    since real messages that land between 10% to 95% confidence are naturally rare.
    """
    bands = [
        (mailbox["p_spam"] < .1, 20), # 14 samples for Inbox
        ((mailbox["p_spam"] >= .1) & (mailbox["p_spam"] < .4), 5), # 5 samples for Review folder (Likely a customer)
        ((mailbox["p_spam"] >= .4) & (mailbox["p_spam"] < .75), 5), # 5 samples for Review folder (Genuinely uncertain)
        ((mailbox["p_spam"] >= .75) & (mailbox["p_spam"] <= .95), 5), # 5 samples for Review folder (Likely Spam)
        (mailbox["p_spam"] > .95, 10), # 10 Samples for Spam folder
    ]
    # Extract the specified quota from each band, combine them, and shuffle the final inbox
    sample = pd.concat([mailbox[mask].sample(min(n, mask.sum()), random_state=42) for mask, n in bands])
    sample = sample.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle "arrival" order

    # fake mail metadata
    # doesn't touch the model nor the routing
    now = pd.Timestamp.now(tz="Europe/Belgrade")
    sample["received_at"] = [now - timedelta(minutes=15 * i) for i in range(len(sample))]
    sample["sender"] = [f"name.surname{i + 1}@iu-study.rs" for i in range(len(sample))]
    sample["subject"] = sample["message"].str.slice(0, 45) + "..."

    folders, review_hours = zip(*sample["p_spam"].map(route_email)) # Applying allocation
    sample["folder"] = folders
    sample["review_by"] = [
        (received + timedelta(hours=h)) if h is not None else None
        for received, h in zip(sample["received_at"], review_hours)
    ]
    return sample


def spam_badge(p):
    """Small color coded confidence readout."""
    if p < 0.2:
        st.success(f"Spam percentage: {p*100:.2f}%")
    elif p > 0.8:
        st.error(f"Spam percentage: {p*100:.2f}%")
    else:
        st.warning(f"Spam percentage: {p*100:.2f}%")


# Initialize the 'mail' list in Streamlit session state if it doesn't already exist
if "mail" not in st.session_state:
    # Load email DataFrame, convert to a list of row dictionaries for easy mutation, and store in session memory
    st.session_state.mail = load_mock_mailbox().to_dict("records")  # list of dict rows, easy to mutate
# Initialize an empty list to record user actions (e.g., moving or approving emails) during this browser session
if "log" not in st.session_state:
    st.session_state.log = []


def move_message(idx, new_folder, log_line):
    st.session_state.mail[idx]["folder"] = new_folder
    st.session_state.log.insert(0, log_line)
    st.rerun()


# Fetch the current list of emails from Streamlit session memory
mail = st.session_state.mail
# Calculate live message counts for each folder using a dictionary comprehension
counts = {f: sum(1 for m in mail if m["folder"] == f) for f in ["Inbox", "Review folder", "Spam"]}

st.sidebar.title("Anri Stepanian's Mailbox")

folder_names = ["Inbox", "Review folder", "Drafts", "Sent", "Spam"]


def format_folder(name):
    return f"{name} ({counts[name]})" if name in counts else name


folder_choice = st.sidebar.radio("Folders", folder_names, format_func=format_folder, key="folder_choice")

st.sidebar.caption("Still in Beta...")

st.title("IU International University of Applied Sciences: Spam Filter")

# Check if the user selected the Inbox folder from the sidebar
if folder_choice == "Inbox":
    st.subheader("Inbox")
    # Filter for all emails currently categorized in the Inbox
    inbox_items = [m for m in mail if m["folder"] == "Inbox"]
    # Display empty state text if no messages exist in the inbox
    if not inbox_items:
        st.caption("Inbox is empty.")
    # Sort messages by timestamp and render each email card
    for m in sorted(inbox_items, key=lambda x: x["received_at"], reverse=True):
        # Create a 4:1 column ratio (80% for sender/subject, 20% for timestamp)
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"**{m['sender']}** {m['subject']}")
            c2.caption(m["received_at"].strftime("%d/%m %H:%M"))
            # Display the email body text beneath the header
            st.caption(m["message"])

# Check if the user selected the Review folder from the sidebar navigation
elif folder_choice == "Review folder":
    st.subheader("Review folder")
    st.caption("Sorted by review-by-time deadline")
    review_items = [(i, m) for i, m in enumerate(mail) if m["folder"] == "Review folder"]
    # Sort the items by their review-by-time deadline
    review_items.sort(key=lambda pair: pair[1]["review_by"])
    # Display empty state text if no messages require review
    if not review_items:
        st.caption("Nothing waiting for review.")
    # Render a card for each email needing human review
    for idx, m in review_items:
        with st.container(border=True):
            # Header layout: 80% for Sender + Subject, 20% for Review Deadline
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"**{m['sender']}** {m['subject']}")
            c2.caption("Review by\n" + m["review_by"].strftime("%d/%m %H:%M"))
            # Display email body text and the spam probability badge
            st.caption(m["message"])
            spam_badge(m["p_spam"])
            # Render action buttons side-by-side
            b1, b2 = st.columns(2)
            if b1.button("Move to Inbox", key=f"approve_{idx}"):
                move_message(idx, "Inbox", f"Released to Inbox: \"{m['subject']}\"")
            if b2.button("Confirm spam", key=f"spam_{idx}"):
                move_message(idx, "Spam", f"Confirmed spam: \"{m['subject']}\" (feeds retraining)")

# Check if the user selected the Drafts folder from the sidebar
elif folder_choice == "Drafts":
    st.subheader("Drafts")
    # Render an informational callout indicating that email composition is not implemented
    st.info("Composing isn't implemented yet - this folder is a placeholder for now.")

# Check if the user selected the Sent folder from the sidebar
elif folder_choice == "Sent":
    st.subheader("Sent")
    # Render an informational callout indicating that sent mail tracking is not implemented
    st.info("Sending isn't implemented yet - this folder is a placeholder for now.")

# Check if the user selected the Spam folder from the sidebar navigation
elif folder_choice == "Spam":
    st.subheader("Spam")
    spam_items = [(i, m) for i, m in enumerate(mail) if m["folder"] == "Spam"]
    # Display empty state text if no messages exist in the spam folder
    if not spam_items:
        st.caption("Spam folder is empty.")
    # Render a card for each message classified as spam
    for idx, m in spam_items:
        with st.container(border=True):
            # Header layout: 80% for Sender + Subject, 20% for Received Time
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"**{m['sender']}** {m['subject']}")
            c2.caption(m["received_at"].strftime("%d/%m %H:%M"))
            # Display email body text and model confidence score
            st.caption(m["message"])
            spam_badge(m["p_spam"])
            # Render button to handle false positives and move message back to Inbox
            if st.button("Not spam - move to Inbox", key=f"notspam_{idx}"):
                move_message(idx, "Inbox", f"Corrected: moved to Inbox: \"{m['subject']}\"")

# Render a visual divider line to separate main content from the audit log
st.divider()
st.subheader("Recent decisions this session")
# Check if any user actions have been recorded in the session log list
if st.session_state.log:
    # Display up to the 10 most recent log entries as bullet points
    for entry in st.session_state.log[:10]:
        st.write("- " + entry)
else:
    # Render placeholder caption if no actions have been taken yet
    st.caption("Nothing decided yet this session.")