import os
import sys
from flask import Flask, render_template, request, redirect, url_for, flash, session
from github import Github
from dotenv import load_dotenv
from datetime import datetime
import logging

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO_NAME = os.getenv("GITHUB_REPO") # Format: owner/repo-name
GITHUB_BRANCH = "main" # Hardcoded as requested

# Web UI Authentication
WEB_UI_USERNAME = os.getenv("WEB_UI_USERNAME")
WEB_UI_PASSWORD = os.getenv("WEB_UI_PASSWORD")

# Flask Secret Key for sessions (Crucial for security!)
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    print("Error: SECRET_KEY not set in .env. Please generate one.", file=sys.stderr)
    sys.exit(1)

# --- Initialize Flask App ---
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY # Configure the secret key

# Configure logging for Flask
app.logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)


# --- GitHub Initialization ---
if not GITHUB_TOKEN or not GITHUB_REPO_NAME:
    app.logger.error("GITHUB_TOKEN or GITHUB_REPO not set in .env.")
    sys.exit(1)

g = None # Initialize GitHub object
repo = None # Initialize repo object
try:
    g = Github(GITHUB_TOKEN)
    # Test connection and repo access
    repo = g.get_repo(GITHUB_REPO_NAME)
    app.logger.info(f"Successfully connected to GitHub and repository: {GITHUB_REPO_NAME}")
except Exception as e:
    app.logger.error(f"Error connecting to GitHub or accessing repository: {e}")
    app.logger.error("Please check your GITHUB_TOKEN and GITHUB_REPO in the .env file.")

    pass 


# --- Authentication Decorator ---
def login_required(view):
    """Decorator to protect views that require login."""
    import functools
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if 'logged_in' not in session or not session['logged_in']:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for('login'))
        # Check if repo was initialized successfully
        if repo is None:
             flash("GitHub repository not accessible. Please check backend logs and .env file.", "danger")

             return render_template('index.html', repo_name=GITHUB_REPO_NAME, branch_name=GITHUB_BRANCH, recent_commits=None, current_head_commit=None)

        return view(**kwargs)
    return wrapped_view

# --- Helper Functions ---
def get_recent_commits(repo, branch_name, count=20): # Increased default count for display
    """
    Fetches recent commits for a branch.
    Limits the number of commits by iterating the PaginatedList.
    """
    if repo is None:
        return None # Cannot fetch if repo isn't initialized

    try:
        branch = repo.get_branch(branch_name)
        # Get commits starting from the branch head.
        all_commits_iterator = repo.get_commits(sha=branch.commit.sha)

        # Collect only the first 'count' commits from the iterator
        recent_commits_list = []
        for i, commit in enumerate(all_commits_iterator):
            if i >= count:
                break
            recent_commits_list.append(commit)

        return recent_commits_list

    except Exception as e:
        app.logger.error(f"Error fetching commits for branch {branch_name}: {e}")
        flash(f"Error fetching recent commits: {e}", "danger")
        return None

def perform_rollback_to_commit(repo, target_commit_sha, branch_name):
    """
    Performs a rollback by creating a new commit with the target state
    and updating the branch reference. This is NOT a git revert,
    but applies the target commit's tree as a new commit on HEAD.
    """
    if repo is None:
         return False, "Repository not accessible for rollback."

    try:
        app.logger.info(f"Attempting rollback for {branch_name} to commit {target_commit_sha}")

        # 1. Get the target commit object (the state we want to revert to)
        target_commit = repo.get_commit(sha=target_commit_sha)
        target_tree_sha = target_commit.commit.tree.sha # Access tree via .commit attribute
        app.logger.info(f"Target tree SHA: {target_tree_sha}")

        # 2. Get the current branch and its HEAD commit
        current_branch = repo.get_branch(branch_name)
        current_head_commit = current_branch.commit
        current_head_commit_sha = current_head_commit.sha
        app.logger.info(f"Current HEAD SHA: {current_head_commit_sha}")

        if target_commit_sha == current_head_commit_sha:
             app.logger.info("Target commit is already the current HEAD.")
             return False, "Target commit is already the current HEAD."

        # 3. Get the GitCommit object for the current HEAD to use as the parent of the new commit
        current_git_head_commit = repo.get_git_commit(sha=current_head_commit_sha)
        app.logger.info(f"Current HEAD GitCommit SHA for parent: {current_git_head_commit.sha}")

        # 4. Get the GitTree object for the target state
        target_tree = repo.get_git_tree(target_tree_sha)
        app.logger.info(f"Target GitTree SHA: {target_tree.sha}")


        # 5. Create a new Git Commit object using the correct method

        commit_message = f"Rollback: Revert branch to state of commit {target_commit_sha[:7]} via web UI"
        app.logger.info(f"Creating new Git commit with message: '{commit_message}'")

        new_git_commit = repo.create_git_commit(
            message=commit_message,
            tree=target_tree, 
            parents=[current_git_head_commit] 
        )
        app.logger.info(f"Successfully created new Git commit with SHA: {new_git_commit.sha}")

        # 6. Get the reference for the branch (e.g., 'refs/heads/main')
        ref_path = f'heads/{branch_name}'
        app.logger.info(f"Updating reference '{ref_path}' to point to {new_git_commit.sha}")
        ref = repo.get_git_ref(ref_path)

        # 7. Update the reference to point to the new commit's SHA
        ref.edit(new_git_commit.sha)
        app.logger.info("Reference update successful.")

        return True, f"Successfully rolled back to state of commit {target_commit_sha[:7]}. New commit SHA: {new_git_commit.sha[:7]}"

    except Exception as e:
        # Log the full error for debugging
        app.logger.error(f"Rollback Error for commit {target_commit_sha}: {e}")

        return False, f"Error during rollback: {e}"

# --- Routes ---

@app.route('/')
@login_required # Protected route
def index():
    # repo is guaranteed to be not None by the login_required decorator check
    recent_commits = get_recent_commits(repo, GITHUB_BRANCH, count=20)

    current_head_commit = None
    try:
        current_head_commit = repo.get_branch(GITHUB_BRANCH).commit
    except Exception as e:
         app.logger.error(f"Error getting current HEAD for branch {GITHUB_BRANCH}: {e}")

         if recent_commits is not None: 
             flash(f"Error getting current HEAD information: {e}", "danger")


    return render_template(
        'index.html',
        repo_name=GITHUB_REPO_NAME,
        branch_name=GITHUB_BRANCH,
        recent_commits=recent_commits,
        current_head_commit=current_head_commit
    )

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == WEB_UI_USERNAME and password == WEB_UI_PASSWORD:
            session['logged_in'] = True
            flash("Login successful!", "success")
            app.logger.info(f"User '{username}' logged in successfully.")
            return redirect(url_for('index'))
        else:
            flash("Invalid credentials.", "danger")
            app.logger.warning(f"Failed login attempt for user '{username}'.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    app.logger.info("User logged out.")
    session.pop('logged_in', None)
    flash("Logged out.", "info")
    return redirect(url_for('login'))


@app.route('/rollback', methods=['POST'])
@login_required 
def rollback():

    rollback_type = request.form.get('rollback_type')
    target_commit_sha = None

    app.logger.info(f"Received rollback request: type={rollback_type}")

    try:
        # Get current HEAD before deciding target commit
        current_branch = repo.get_branch(GITHUB_BRANCH)
        current_head_commit = current_branch.commit
        current_head_commit_sha = current_head_commit.sha

        if rollback_type == 'previous':
            if not current_head_commit.parents:
                 flash("The current commit has no parents (it might be the initial commit). Cannot roll back to previous.", "warning")
                 app.logger.warning("Rollback attempt to previous failed: Initial commit has no parent.")
                 return redirect(url_for('index'))
            target_commit_sha = current_head_commit.parents[0].sha
            flash(f"Attempting rollback to previous commit: {target_commit_sha[:7]}", "info")
            app.logger.info(f"Target commit determined (previous): {target_commit_sha}")


        elif rollback_type == 'specific':
            target_commit_sha = request.form.get('specific_commit_sha', '').strip()
            if not target_commit_sha:
                flash("Please provide a specific commit hash.", "warning")
                app.logger.warning("Rollback attempt to specific failed: No commit hash provided.")
                return redirect(url_for('index'))

            # Basic SHA format check (optional but good) - a full 40 char check is better
            if len(target_commit_sha) < 7 or not all(c in '0123456789abcdefABCDEF' for c in target_commit_sha):
                 flash("Invalid commit hash format.", "warning")
                 app.logger.warning(f"Rollback attempt to specific failed: Invalid hash format '{target_commit_sha}'.")
                 return redirect(url_for('index'))

            flash(f"Attempting rollback to specific commit: {target_commit_sha[:7]}", "info")
            app.logger.info(f"Target commit determined (specific): {target_commit_sha}")

            # Verify the specific commit exists before proceeding
            try:
                 repo.get_commit(sha=target_commit_sha)
                 app.logger.info(f"Validated specific target commit exists: {target_commit_sha}")
            except Exception as e:
                 flash(f"Specific target commit '{target_commit_sha[:7]}' not found in the repository history.", "danger")
                 app.logger.warning(f"Specific target commit validation failed: {target_commit_sha} - {e}")
                 return redirect(url_for('index'))


        else:
            flash("Invalid rollback type.", "danger")
            app.logger.error(f"Rollback attempt failed: Invalid rollback type '{rollback_type}'.")
            return redirect(url_for('index'))

        # Perform the rollback using the determined target_commit_sha
        if target_commit_sha:
            success, message = perform_rollback_to_commit(repo, target_commit_sha, GITHUB_BRANCH)
            if success:
                flash(message, "success")
                app.logger.info(f"Rollback successful: {message}")
            else:
                flash(message, "danger") # Message already contains error info
                app.logger.error(f"Rollback failed: {message}")


    except Exception as e:
        flash(f"An unexpected error occurred during rollback setup: {e}", "danger")
        # Log the error server-side as well
        app.logger.error(f"Unexpected error during rollback setup: {e}")


    return redirect(url_for('index'))

# --- Run the App ---
if __name__ == '__main__':

    print("\n--- IMPORTANT SECURITY WARNING ---")
    print("Running with debug=True and host='0.0.0.0' is NOT safe for production.")
    print("This setup exposes the debugger and makes the app publicly accessible.")
    print("Use a production WSGI server (like Gunicorn) in a real deployment.")
    print("---------------------------------\n")


    if repo is None:
         app.logger.error("Repository initialization failed. The app may not function correctly.")

         pass 

    app.run(debug=True, host='0.0.0.0', port=5000)
