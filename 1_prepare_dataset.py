"""
============================================================
STEP 1: DATASET PREPARATION
============================================================
Build a labeled dataset of (user_story → test_scenarios)
pairs. We include:
  - Hand-crafted seed examples (domain: e-commerce, auth, forms)
  - Augmentation via prompt templates
  - Train/val/test split (80/10/10)
  - Export to HuggingFace datasets format
============================================================
"""

import json
import os
import random
from datasets import Dataset, DatasetDict

# ──────────────────────────────────────────────────────────
# SEED DATA: (user_story, test_scenarios) pairs
# Add as many as you have. ~50+ is minimum; 200+ is good.
# ──────────────────────────────────────────────────────────
RAW_DATA = [
    {
        "user_story": (
            "As a registered user, I want to log in with my email and password "
            "so that I can access my account dashboard."
        ),
        "test_scenarios": (
            "1. Valid login with correct email and password → redirects to dashboard\n"
            "2. Login with incorrect password → show error 'Invalid credentials'\n"
            "3. Login with unregistered email → show error 'Account not found'\n"
            "4. Login with empty email field → show validation error 'Email is required'\n"
            "5. Login with empty password field → show validation error 'Password is required'\n"
            "6. Login with SQL injection in email field → system rejects, no crash\n"
            "7. Login after account lockout (5 failed attempts) → show lockout message\n"
            "8. Login with valid credentials but session already active → handle gracefully\n"
            "9. 'Remember me' checkbox selected → session persists after browser close\n"
            "10. Forgot password link → redirects to password reset page"
        ),
    },
    {
        "user_story": (
            "As a shopper, I want to add items to my cart so that I can "
            "purchase multiple products in one transaction."
        ),
        "test_scenarios": (
            "1. Add single in-stock item → item appears in cart with correct quantity\n"
            "2. Add same item twice → cart shows combined quantity\n"
            "3. Add out-of-stock item → 'Add to cart' button disabled, tooltip shown\n"
            "4. Add item at maximum stock limit (e.g., last 1 unit) → success, quantity capped\n"
            "5. Add item exceeding stock quantity → show 'Only N items available'\n"
            "6. Cart icon badge updates correctly after each addition\n"
            "7. Cart persists across page navigation\n"
            "8. Cart persists after page refresh\n"
            "9. Guest user cart survives browser session until checkout\n"
            "10. Cart merges with saved cart when guest logs in"
        ),
    },
    {
        "user_story": (
            "As an admin, I want to reset a user's password so that I can "
            "help users who are locked out of their accounts."
        ),
        "test_scenarios": (
            "1. Admin resets password for valid user → success message shown, user receives email\n"
            "2. Admin resets password for non-existent user ID → show error 'User not found'\n"
            "3. Admin without reset_password permission attempts reset → access denied (403)\n"
            "4. Reset email is sent with valid reset link\n"
            "5. Reset link expires after 24 hours → show 'Link expired' on click\n"
            "6. Reset link can only be used once → second use shows error\n"
            "7. New password must meet complexity rules → enforce minimum 8 chars, 1 number\n"
            "8. Confirm password mismatch → show validation error\n"
            "9. Audit log records admin who triggered the reset\n"
            "10. User receives notification email after password is changed"
        ),
    },
    {
        "user_story": (
            "As a user, I want to upload a profile picture so that other "
            "users can identify me easily."
        ),
        "test_scenarios": (
            "1. Upload valid JPEG under 5MB → image saved, preview displayed\n"
            "2. Upload valid PNG under 5MB → success\n"
            "3. Upload file exceeding 5MB → show error 'File too large (max 5MB)'\n"
            "4. Upload unsupported file type (PDF, exe) → show error 'Invalid file type'\n"
            "5. Upload image with very small dimensions (10x10px) → warn user about quality\n"
            "6. Profile picture visible on user's public profile page\n"
            "7. Old profile picture is replaced (not duplicated) on re-upload\n"
            "8. Upload with no file selected → button disabled or validation shown\n"
            "9. Upload during slow network → loading spinner displayed\n"
            "10. Uploaded image is resized/compressed to standard dimensions"
        ),
    },
    {
        "user_story": (
            "As a customer, I want to search for products by keyword so that "
            "I can quickly find what I'm looking for."
        ),
        "test_scenarios": (
            "1. Search with exact product name → correct product appears first\n"
            "2. Search with partial keyword → relevant results shown\n"
            "3. Search with misspelled keyword → spelling suggestions shown\n"
            "4. Search with no results → 'No products found' message with suggestions\n"
            "5. Search with empty query → no search executed or show all products\n"
            "6. Search is case-insensitive\n"
            "7. Search results are paginated (max 20 per page)\n"
            "8. Search result count matches total found items\n"
            "9. Search with special characters (e.g., & < >) → handled safely, no crash\n"
            "10. Search performance: results returned within 2 seconds"
        ),
    },
    {
        "user_story": (
            "As a user, I want to receive email notifications for order status updates "
            "so that I am always informed about my purchase."
        ),
        "test_scenarios": (
            "1. Order placed → confirmation email sent within 1 minute\n"
            "2. Order shipped → email sent with tracking number\n"
            "3. Order delivered → delivery confirmation email sent\n"
            "4. Order cancelled → cancellation email with reason\n"
            "5. Email contains correct order ID, items, total price\n"
            "6. User unsubscribes from notifications → no further emails sent\n"
            "7. Notification preference toggle works (on/off in settings)\n"
            "8. Email sent to correct registered email address\n"
            "9. Email renders correctly on mobile clients\n"
            "10. No duplicate emails for same status event"
        ),
    },
    {
        "user_story": (
            "As a manager, I want to view a dashboard of team performance metrics "
            "so that I can track progress against quarterly goals."
        ),
        "test_scenarios": (
            "1. Dashboard loads within 3 seconds with all KPI widgets\n"
            "2. Metrics display correct values matching underlying data\n"
            "3. Date range filter updates all charts and numbers\n"
            "4. Drill-down on KPI card shows individual contributor breakdown\n"
            "5. Export to CSV contains all visible data\n"
            "6. Dashboard is read-only for non-manager roles\n"
            "7. Refresh button fetches latest data without full page reload\n"
            "8. Empty state shown when no data for selected period\n"
            "9. Charts are accessible (ARIA labels, keyboard navigation)\n"
            "10. Dashboard layout is responsive on tablet screen size"
        ),
    },
    {
        "user_story": (
            "As a user, I want to reset my password via email so that "
            "I can regain access if I forget my credentials."
        ),
        "test_scenarios": (
            "1. Valid email submitted → reset email sent, success message shown\n"
            "2. Unregistered email submitted → generic message (no user enumeration)\n"
            "3. Reset link clicked → redirects to password reset form\n"
            "4. New valid password submitted → password updated, user logged in\n"
            "5. Reset link expires after 1 hour → show 'Link expired'\n"
            "6. Used reset link clicked again → show 'Link already used'\n"
            "7. Weak password rejected → show complexity requirements\n"
            "8. Confirm password mismatch → inline validation error\n"
            "9. Multiple reset requests → only latest link is valid\n"
            "10. After reset, all other sessions are invalidated"
        ),
    },
    {
        "user_story": (
            "As a developer, I want to integrate a payment gateway so that "
            "customers can complete purchases securely."
        ),
        "test_scenarios": (
            "1. Valid card details → payment processed, order confirmed\n"
            "2. Expired card → show 'Card expired' error\n"
            "3. Insufficient funds → show 'Payment declined'\n"
            "4. Invalid CVV → show 'Invalid security code'\n"
            "5. Network timeout during payment → retry mechanism, no duplicate charge\n"
            "6. Payment data transmitted over HTTPS only\n"
            "7. Card details are not stored in plain text in database\n"
            "8. Successful payment generates unique transaction ID\n"
            "9. Refund process reverses the correct amount\n"
            "10. Payment gateway downtime → graceful error, no data loss"
        ),
    },
    {
        "user_story": (
            "As a content editor, I want to schedule posts to be published at a future time "
            "so that I can plan content in advance."
        ),
        "test_scenarios": (
            "1. Schedule post for future date/time → post not visible until that time\n"
            "2. Scheduled time in the past → validation error 'Cannot schedule in the past'\n"
            "3. Post publishes automatically at scheduled time\n"
            "4. Scheduled post can be edited before publish time\n"
            "5. Scheduled post can be cancelled (reverted to draft)\n"
            "6. Timezone is respected for scheduled time\n"
            "7. System notification sent to editor when post publishes\n"
            "8. Calendar view shows all scheduled posts correctly\n"
            "9. Two posts scheduled at same time → both publish without conflict\n"
            "10. Server downtime at scheduled time → post publishes on recovery"
        ),
    },
    {
        "user_story": (
            "As a mobile user, I want the app to work offline so that "
            "I can access key features without internet connection."
        ),
        "test_scenarios": (
            "1. App launches offline → shows cached content from last sync\n"
            "2. Actions taken offline are queued and synced on reconnect\n"
            "3. Create item offline → item appears locally, syncs when online\n"
            "4. Edit item offline → changes synced without data loss\n"
            "5. Delete item offline → deletion reflected after sync\n"
            "6. Conflict resolution: offline edit vs server edit → user prompted\n"
            "7. Offline indicator banner shown in UI\n"
            "8. Features requiring real-time data disabled offline\n"
            "9. App does not crash on complete network loss\n"
            "10. Sync completes within 5 seconds of reconnection"
        ),
    },
    {
        "user_story": (
            "As a user, I want to filter products by price range "
            "so that I can find items within my budget."
        ),
        "test_scenarios": (
            "1. Set min price only → shows all products above min\n"
            "2. Set max price only → shows all products below max\n"
            "3. Set valid range (e.g., $10-$50) → only products in range shown\n"
            "4. Min price greater than max price → show validation error\n"
            "5. Filter with no products in range → show empty state message\n"
            "6. Filter applies without full page reload\n"
            "7. Price filter combines correctly with category filter\n"
            "8. Filter values persist when navigating back to results\n"
            "9. Clear filter button resets to all products\n"
            "10. Filter handles decimal prices correctly (e.g., $9.99)"
        ),
    },
]


def prepare_for_training(examples):
    """
    Format each example into a prompt-completion pair.
    Flan-T5 style: instruction prefix + input → target output
    """
    formatted = []
    for ex in examples:
        formatted.append({
            "input_text": (
                "Generate comprehensive test scenarios for the following user story:\n\n"
                f"{ex['user_story']}\n\n"
                "Test Scenarios:"
            ),
            "target_text": ex["test_scenarios"],
            "user_story": ex["user_story"],
            "test_scenarios": ex["test_scenarios"],
        })
    return formatted


def split_dataset(data, train_ratio=0.8, val_ratio=0.1):
    random.seed(42)
    random.shuffle(data)
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return data[:train_end], data[train_end:val_end], data[val_end:]


def main():
    os.makedirs("data", exist_ok=True)

    formatted = prepare_for_training(RAW_DATA)
    train_data, val_data, test_data = split_dataset(formatted)

    print(f"📊 Dataset split:")
    print(f"   Train : {len(train_data)} examples")
    print(f"   Val   : {len(val_data)} examples")
    print(f"   Test  : {len(test_data)} examples")

    # Save as HuggingFace DatasetDict
    dataset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
        "test": Dataset.from_list(test_data),
    })
    dataset.save_to_disk("data/test_scenario_dataset")
    print("✅ Dataset saved to data/test_scenario_dataset")

    # Also save raw JSON for inspection
    with open("data/train.json", "w") as f:
        json.dump(train_data, f, indent=2)
    with open("data/val.json", "w") as f:
        json.dump(val_data, f, indent=2)
    with open("data/test.json", "w") as f:
        json.dump(test_data, f, indent=2)

    print("\n📄 Sample training example:")
    print("-" * 60)
    print("INPUT :", formatted[0]["input_text"][:200])
    print("TARGET:", formatted[0]["target_text"][:200])

    return dataset


if __name__ == "__main__":
    main()