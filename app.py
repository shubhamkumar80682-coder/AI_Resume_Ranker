from __future__ import annotations
import logging
import os
from flask import Flask, flash, redirect, render_template, request, session, url_for
import pdfplumber

from db import init_db, create_user, authenticate_user, get_user_by_id
from model import rank_resumes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24).hex())

ALLOWED_EXTENSIONS = {"pdf"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file):
    try:
        file.seek(0)
        text_parts = []
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        extracted = "\n".join(text_parts).strip()
        if not extracted:
            return None, "Could not extract text (PDF may be image-based or encrypted)"
        return extracted, None
    except Exception as e:
        logger.error("PDF extraction failed: %s", e)
        return None, f"Failed to read PDF: {e}"

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ---------------------------------------------------------------------------
# Public routes
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        user, error = authenticate_user(email, password)
        if error:
            flash(error, "error")
            return render_template("login.html")
        session["user_id"] = user["id"]
        session["user_email"] = user["email"]
        flash("Welcome back!", "success")
        return redirect(url_for("dashboard"))
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        if password != confirm:
            flash("Passwords do not match", "error")
            return render_template("register.html")
        user_id, error = create_user(email, password)
        if error:
            flash(error, "error")
            return render_template("register.html")
        session["user_id"] = user_id
        session["user_email"] = email
        flash("Account created successfully!", "success")
        return redirect(url_for("dashboard"))
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))

# ---------------------------------------------------------------------------
# Protected routes
# ---------------------------------------------------------------------------

@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    ranked = []
    if request.method == "POST":
        job_desc = request.form.get("job", "").strip()
        if not job_desc:
            flash("Please provide a job description.", "error")
            return redirect(url_for("dashboard"))
        files = request.files.getlist("resumes")
        valid_files = []
        invalid_files = []
        for file in files:
            if not file or not file.filename:
                continue
            if not allowed_file(file.filename):
                invalid_files.append(file.filename)
                continue
            valid_files.append(file)
        if invalid_files:
            flash(f"Skipped non-PDF files: {', '.join(invalid_files)}", "warning")
        if not valid_files:
            flash("Please upload at least one valid PDF file.", "error")
            return redirect(url_for("dashboard"))
        resumes_text = []
        filenames = []
        for file in valid_files:
            text, error = extract_text_from_pdf(file)
            if error:
                flash(f"{file.filename}: {error}", "error")
                continue
            resumes_text.append(text)
            filenames.append(file.filename)
        if not resumes_text:
            flash("No resumes could be processed.", "error")
            return redirect(url_for("dashboard"))
        try:
            scores = rank_resumes(job_desc, resumes_text)
            ranked = sorted(zip(filenames, scores), key=lambda x: x[1], reverse=True)
            logger.info("User %s ranked %d resumes", session["user_email"], len(ranked))
            flash(f"Successfully ranked {len(ranked)} resume(s).", "success")
        except Exception as e:
            logger.error("Ranking failed: %s", e)
            flash(f"Ranking failed: {e}", "error")
    return render_template("dashboard.html", ranked=ranked)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_db()
    from waitress import serve
    logger.info("Starting server on http://127.0.0.1:5000")
    serve(app, host="127.0.0.1", port=5000)