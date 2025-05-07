document.addEventListener("DOMContentLoaded", function () {
  // DOM Elements - File Upload
  const fileInput = document.getElementById("file-input");
  const uploadArea = document.getElementById("upload-area");
  const fileList = document.getElementById("file-list");
  const uploadBtn = document.getElementById("upload-btn");
  const uploadLoader = document.getElementById("upload-loader");
  const uploadStatus = document.getElementById("upload-status");
  const summaryContent = document.getElementById("summary-content");

  // DOM Elements - Student Profile
  const profileSection = document.getElementById("profile-section");
  const profileForm = document.getElementById("profile-form");
  const profileDisplay = document.getElementById("profile-display");
  const studentNameInput = document.getElementById("student-name");
  const createProfileBtn = document.getElementById("create-profile-btn");
  const profileName = document.getElementById("profile-name");
  const profileLearningStyle = document.getElementById(
    "profile-learning-style"
  );
  const profileKnowledgeLevel = document.getElementById(
    "profile-knowledge-level"
  );
  const profileStrengths = document.getElementById("profile-strengths");
  const profileAreas = document.getElementById("profile-areas");

  // DOM Elements - Learning Journey Progression
  const progressionSection = document.getElementById("progression-section");
  const nextStepBtn = document.getElementById("next-step-btn");
  const progressionSteps = document.querySelectorAll(
    ".progression-steps .step"
  );

  // DOM Elements - Learning Style Assessment
  const learningStyleSection = document.getElementById(
    "learning-style-section"
  );
  const surveyQuestions = document.getElementById("survey-questions");
  const submitSurveyBtn = document.getElementById("submit-survey-btn");
  const surveyResult = document.getElementById("survey-result");
  const learningStyleResult = document.getElementById("learning-style-result");

  // DOM Elements - Pre-Test
  const pretestSection = document.getElementById("pretest-section");
  const pretestTitle = document.getElementById("pretest-title");
  const pretestDescription = document.getElementById("pretest-description");
  const pretestQuestions = document.getElementById("pretest-questions");
  const submitPretestBtn = document.getElementById("submit-pretest-btn");
  const pretestResult = document.getElementById("pretest-result");
  const pretestScore = document.getElementById("pretest-score");
  const knowledgeLevel = document.getElementById("knowledge-level");

  // DOM Elements - Learning Path
  const learningPathSection = document.getElementById("learning-path-section");
  const learningPathTitle = document.getElementById("learning-path-title");
  const learningPathDescription = document.getElementById(
    "learning-path-description"
  );
  const learningObjectivesList = document.getElementById(
    "learning-objectives-list"
  );
  const modulesList = document.getElementById("modules-list");

  // DOM Elements - Module Content
  const moduleContentSection = document.getElementById(
    "module-content-section"
  );
  const backToModulesBtn = document.getElementById("back-to-modules-btn");
  const moduleTitle = document.getElementById("module-title");
  const moduleDescription = document.getElementById("module-description");
  const moduleContent = document.getElementById("module-content");
  const startDiscussionBtn = document.getElementById("start-discussion-btn");
  const takePosttestBtn = document.getElementById("take-posttest-btn");

  // DOM Elements - Peer Discussion
  const discussionSection = document.getElementById("discussion-section");
  const discussionMessages = document.getElementById("discussion-messages");
  const discussionInput = document.getElementById("discussion-input");
  const sendDiscussionBtn = document.getElementById("send-discussion-btn");
  const discussionLoader = document.getElementById("discussion-loader");
  const endDiscussionBtn = document.getElementById("end-discussion-btn");

  // DOM Elements - Post-Test
  const posttestSection = document.getElementById("posttest-section");
  const posttestTitle = document.getElementById("posttest-title");
  const posttestDescription = document.getElementById("posttest-description");
  const posttestQuestions = document.getElementById("posttest-questions");
  const submitPosttestBtn = document.getElementById("submit-posttest-btn");
  const posttestResult = document.getElementById("posttest-result");
  const posttestScore = document.getElementById("posttest-score");
  const newKnowledgeLevel = document.getElementById("new-knowledge-level");
  const levelChangeMessage = document.getElementById("level-change-message");

  // DOM Elements - Learning Log
  const learningLogSection = document.getElementById("learning-log-section");
  const learningLogContent = document.getElementById("learning-log-content");
  const submitLogBtn = document.getElementById("submit-log-btn");
  const logAnalysis = document.getElementById("log-analysis");
  const understandingLevel = document.getElementById("understanding-level");
  const logStrengths = document.getElementById("log-strengths");
  const logAreas = document.getElementById("log-areas");
  const logNextSteps = document.getElementById("log-next-steps");
  const continueBtn = document.getElementById("continue-btn");

  // DOM Elements - Q&A
  const questionInput = document.getElementById("question-input");
  const askBtn = document.getElementById("ask-btn");
  const messages = document.getElementById("messages");
  const chatLoader = document.getElementById("chat-loader");

  // State variables
  let files = [];
  let currentLearningPath = null;
  let currentModuleIndex = 0;
  let currentStep = "profile";
  let currentSurvey = null;
  let currentTopic = "";

  // Initialize the application
  initializeApp();

  // Track the application flow
  let appFlow = {
    profileComplete: false,
    learningStyleComplete: false,
    pdfUploaded: false,
  };

  // Event Listeners
  // Create Profile button
  createProfileBtn.addEventListener("click", function () {
    console.log("Create Profile button clicked");
    const name = studentNameInput.value.trim();
    if (!name) {
      alert("Please enter your name.");
      return;
    }

    // Add loading indication
    createProfileBtn.textContent = "Creating...";
    createProfileBtn.disabled = true;

    createProfile();
  });
  uploadArea.addEventListener("dragover", function (e) {
    e.preventDefault();
    uploadArea.style.borderColor = "#4fc3dc";
    uploadArea.style.backgroundColor = "rgba(79, 195, 220, 0.1)";
  });

  uploadArea.addEventListener("dragleave", function () {
    uploadArea.style.borderColor = "#4fc3dc";
    uploadArea.style.backgroundColor = "";
  });

  uploadArea.addEventListener("drop", function (e) {
    e.preventDefault();
    uploadArea.style.borderColor = "#4fc3dc";
    uploadArea.style.backgroundColor = "";

    handleFiles(e.dataTransfer.files);
  });

  uploadArea.addEventListener("click", function () {
    fileInput.click();
  });

  fileInput.addEventListener("change", function () {
    handleFiles(fileInput.files);
  });

  // Event Listeners
  document.addEventListener("DOMContentLoaded", function () {
    // Initialize the app when DOM is loaded
    initializeApp();

    // Profile section
    const createProfileBtn = document.getElementById("create-profile-btn");
    createProfileBtn.addEventListener("click", function () {
      console.log("Create Profile button clicked");
      createProfile();
    });

    // Other event listeners...
  });

  // Remove the duplicate event listener
  // createProfileBtn.addEventListener("click", function() {
  //   createProfile();
  // });

  // Learning journey progression
  nextStepBtn.addEventListener("click", function () {
    advanceToNextStep();
  });

  // Survey submission
  submitSurveyBtn.addEventListener("click", function () {
    submitLearningStyleSurvey();
  });

  // Pretest submission
  submitPretestBtn.addEventListener("click", function () {
    submitPretest();
  });

  // Module navigation
  backToModulesBtn.addEventListener("click", function () {
    hideAllSections();
    learningPathSection.classList.remove("hidden");
  });

  // Discussion controls
  startDiscussionBtn.addEventListener("click", function () {
    startDiscussion();
  });

  sendDiscussionBtn.addEventListener("click", function () {
    sendDiscussionMessage();
  });

  discussionInput.addEventListener("keypress", function (e) {
    if (e.key === "Enter") {
      sendDiscussionMessage();
    }
  });

  endDiscussionBtn.addEventListener("click", function () {
    endDiscussion();
  });

  // Posttest submission
  submitPosttestBtn.addEventListener("click", function () {
    submitPosttest();
  });

  // Learning log submission
  submitLogBtn.addEventListener("click", function () {
    submitLearningLog();
  });

  // Continue button after learning log
  continueBtn.addEventListener("click", function () {
    hideAllSections();
    learningPathSection.classList.remove("hidden");
  });

  // Upload button
  uploadBtn.addEventListener("click", function () {
    uploadFiles();
  });

  // Q&A functionality
  askBtn.addEventListener("click", askQuestion);
  questionInput.addEventListener("keypress", function (e) {
    if (e.key === "Enter") {
      askQuestion();
    }
  });

  // Helper functions
  function initializeApp() {
    // Hide the upload loader
    uploadLoader.style.display = "none";

    // Hide document sections initially
    document.querySelector(".card:nth-child(3)").style.display = "none"; // Upload card
    document.querySelector(".card:nth-child(4)").style.display = "none"; // Document summary card

    // Ensure profile form is visible initially
    profileForm.style.display = "block";
    profileDisplay.style.display = "none";

    // Hide progression section initially
    progressionSection.style.display = "none";

    // Fetch or create student profile
    fetch("/api/profile")
      .then((response) => response.json())
      .then((data) => {
        // Check if this is an auto-generated profile with default name
        const isAutoGenerated = data.name.includes("Student_");

        if (!isAutoGenerated) {
          // For real user-created profiles, display it
          updateProfileDisplay(data);
        } else {
          // For auto-generated profiles, keep form visible
          profileForm.style.display = "block";
          profileDisplay.style.display = "none";
        }
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  }

  function handleFiles(selectedFiles) {
    for (let i = 0; i < selectedFiles.length; i++) {
      const file = selectedFiles[i];
      if (file.type === "application/pdf") {
        files.push(file);
      }
    }

    updateFileList();
  }

  function updateFileList() {
    fileList.innerHTML = "";

    files.forEach((file, index) => {
      const li = document.createElement("li");
      li.innerHTML = `
                <span>${file.name} (${formatFileSize(file.size)})</span>
                <button class="remove-btn" data-index="${index}">
                    <i class="fas fa-times"></i>
                </button>
            `;
      fileList.appendChild(li);
    });

    uploadBtn.disabled = files.length === 0;

    // Add event listeners to remove buttons
    document.querySelectorAll(".remove-btn").forEach((btn) => {
      btn.addEventListener("click", function () {
        const index = parseInt(this.getAttribute("data-index"));
        files.splice(index, 1);
        updateFileList();
      });
    });
  }

  function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + " bytes";
    else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
    else return (bytes / 1048576).toFixed(1) + " MB";
  }

  function uploadFiles() {
    if (files.length === 0) return;

    const formData = new FormData();
    files.forEach((file) => {
      formData.append("files[]", file);
    });

    uploadLoader.style.display = "block";
    uploadBtn.disabled = true;

    fetch("/upload", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        uploadLoader.style.display = "none";

        if (data.success) {
          uploadStatus.textContent = data.message;
          uploadStatus.className = "status success";
          uploadStatus.style.display = "block";

          summaryContent.innerHTML = marked.parse(data.summary);

          // Enable chat functionality
          questionInput.disabled = false;
          askBtn.disabled = false;

          // Add a system message
          addMessage(
            "Documents processed! I'm ready to answer your questions.",
            "bot"
          );

          // Mark PDFs as uploaded and continue to pretest
          appFlow.pdfUploaded = true;

          // Update the current step
          currentStep = "upload";

          // Show the progression section to continue the learning journey
          progressionSection.style.display = "block";
          nextStepBtn.textContent = "Continue to Pre-Test";

          // Update progression UI to show upload complete
          updateProgressionUI();

          // Clear file list
          files = [];
          updateFileList();
        } else {
          uploadStatus.textContent = data.error || "An error occurred";
          uploadStatus.className = "status error";
          uploadStatus.style.display = "block";
          uploadBtn.disabled = false;
        }
      })
      .catch((error) => {
        uploadLoader.style.display = "none";
        uploadStatus.textContent = "Error: " + error.message;
        uploadStatus.className = "status error";
        uploadStatus.style.display = "block";
        uploadBtn.disabled = files.length > 0;
      });
  }

  function createProfile() {
    const name = studentNameInput.value.trim();
    if (!name) {
      alert("Please enter your name.");
      return;
    }

    // Add visual feedback
    createProfileBtn.textContent = "Creating...";
    createProfileBtn.disabled = true;

    console.log("Creating profile for: " + name);

    fetch("/api/profile", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ name: name }),
    })
      .then((response) => {
        console.log("Response received:", response);
        if (!response.ok) {
          throw new Error("Server returned " + response.status);
        }
        return response.json();
      })
      .then((data) => {
        console.log("Profile created:", data);
        updateProfileDisplay(data);

        // Show progression section after profile creation
        progressionSection.style.display = "block";

        // Hide the upload section until learning style is complete
        document.querySelector(".card:nth-child(3)").style.display = "none"; // Upload card

        // Update current step
        currentStep = "profile";

        // Update progression UI
        updateProgressionUI();

        // Reset button
        createProfileBtn.textContent = "Create Profile";
        createProfileBtn.disabled = false;
      })
      .catch((error) => {
        console.error("Error creating profile:", error);
        alert("Error creating profile. Please try again.");

        // Reset button
        createProfileBtn.textContent = "Create Profile";
        createProfileBtn.disabled = false;
      });
  }

  function updateProfileDisplay(profile) {
    // Check if this is an auto-generated profile (has default name with underscore)
    const isAutoGenerated = profile.name.includes("Student_");

    if (isAutoGenerated) {
      // For auto-generated profiles, keep showing the form
      profileForm.style.display = "block";
      profileDisplay.style.display = "none";

      // Set the input field to empty for better user experience
      studentNameInput.value = "";
      studentNameInput.placeholder = "Enter your name";

      // Don't show progression yet
      progressionSection.style.display = "none";
    } else {
      // For user-created profiles, show the profile display
      // Update profile display
      profileName.textContent = profile.name;
      profileLearningStyle.textContent =
        profile.learning_style || "Not assessed yet";
      profileKnowledgeLevel.textContent =
        profile.current_knowledge_level || "Not assessed yet";

      // Update strengths and areas for improvement
      profileStrengths.innerHTML = "";
      if (profile.strengths && profile.strengths.length > 0) {
        profile.strengths.forEach((strength) => {
          const li = document.createElement("li");
          li.textContent = strength;
          profileStrengths.appendChild(li);
        });
      } else {
        const li = document.createElement("li");
        li.textContent = "No strengths identified yet";
        profileStrengths.appendChild(li);
      }

      profileAreas.innerHTML = "";
      if (
        profile.areas_for_improvement &&
        profile.areas_for_improvement.length > 0
      ) {
        profile.areas_for_improvement.forEach((area) => {
          const li = document.createElement("li");
          li.textContent = area;
          profileAreas.appendChild(li);
        });
      } else {
        const li = document.createElement("li");
        li.textContent = "No areas for improvement identified yet";
        profileAreas.appendChild(li);
      }

      // Show profile display and hide form
      profileForm.style.display = "none";
      profileDisplay.style.display = "block";

      // Show progression section after profile creation
      progressionSection.style.display = "block";

      // Hide the upload section until learning style is complete
      document.querySelector(".card:nth-child(3)").style.display = "none"; // Upload card

      // Update current step
      currentStep = "profile";

      // Update progression UI
      updateProgressionUI();
    }
  }

  function advanceToNextStep() {
    hideAllSections();

    // Determine the next step based on current state
    switch (currentStep) {
      case "profile":
        // After profile creation, go to learning style assessment
        fetchLearningStyleSurvey();
        appFlow.profileComplete = true;
        break;
      case "learning-style":
        // After learning style assessment, show file upload area
        appFlow.learningStyleComplete = true;
        // Show upload section and hide progression until files are uploaded
        document.querySelector(".card:nth-child(3)").style.display = "block"; // Upload card
        progressionSection.style.display = "none";
        return; // Don't update progression UI yet
      case "upload":
        // After file upload, proceed to pretest
        fetchPretest();
        break;
      case "pretest":
        fetchLearningPath();
        break;
      case "learning-path":
        loadModule(0); // Load the first module
        break;
      default:
        // Default to learning style assessment
        fetchLearningStyleSurvey();
    }

    // Update the progression UI
    updateProgressionUI();
  }

  function hideAllSections() {
    // Hide all content sections
    learningStyleSection.classList.add("hidden");
    pretestSection.classList.add("hidden");
    learningPathSection.classList.add("hidden");
    moduleContentSection.classList.add("hidden");
    discussionSection.classList.add("hidden");
    posttestSection.classList.add("hidden");
    learningLogSection.classList.add("hidden");
  }

  function updateProgressionUI() {
    // Update the progression steps UI
    progressionSteps.forEach((step) => {
      step.classList.remove("active", "completed");

      const stepName = step.getAttribute("data-step");

      if (stepName === currentStep) {
        step.classList.add("active");
      } else if (
        (stepName === "learning-style" &&
          ["upload", "pretest", "learning-path", "modules"].includes(
            currentStep
          )) ||
        (stepName === "pretest" &&
          ["learning-path", "modules"].includes(currentStep)) ||
        (stepName === "learning-path" && currentStep === "modules")
      ) {
        step.classList.add("completed");
      }
    });

    // Update the next step button text
    switch (currentStep) {
      case "profile":
        nextStepBtn.textContent = "Begin Learning Style Assessment";
        break;
      case "learning-style":
        nextStepBtn.textContent = "Continue to Document Upload";
        break;
      case "upload":
        nextStepBtn.textContent = "Take Pre-Test";
        break;
      case "pretest":
        nextStepBtn.textContent = "Generate Learning Path";
        break;
      case "learning-path":
        nextStepBtn.textContent = "Start First Module";
        break;
      case "modules":
        nextStepBtn.textContent = "Continue Learning";
        break;
      default:
        nextStepBtn.textContent = "Continue";
    }
  }

  function fetchLearningStyleSurvey() {
    fetch("/api/learning-style-survey")
      .then((response) => response.json())
      .then((data) => {
        displayLearningStyleSurvey(data);
        currentStep = "learning-style";
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  }

  function displayLearningStyleSurvey(survey) {
    // Store the survey for later submission
    currentSurvey = survey;

    // Clear previous questions
    surveyQuestions.innerHTML = "";

    // Display title and description
    const header = document.createElement("div");
    header.className = "survey-header";
    header.innerHTML = `
      <h3>${survey.title}</h3>
      <p>${survey.description}</p>
    `;
    surveyQuestions.appendChild(header);

    // Create questions
    survey.questions.forEach((question, index) => {
      const questionDiv = document.createElement("div");
      questionDiv.className = "question";

      const questionText = document.createElement("p");
      questionText.className = "question-text";
      questionText.textContent = `${index + 1}. ${question.question}`;
      questionDiv.appendChild(questionText);

      const choicesDiv = document.createElement("div");
      choicesDiv.className = "choices";

      question.choices.forEach((choice, choiceIndex) => {
        const label = document.createElement("label");
        label.className = "choice-label";

        const radio = document.createElement("input");
        radio.type = "radio";
        radio.name = `question-${index}`;
        radio.value = String.fromCharCode(65 + choiceIndex); // A, B, C, etc.

        label.appendChild(radio);
        label.appendChild(document.createTextNode(` ${choice}`));
        choicesDiv.appendChild(label);
      });

      questionDiv.appendChild(choicesDiv);
      surveyQuestions.appendChild(questionDiv);
    });

    // Show the section
    hideAllSections();
    learningStyleSection.classList.remove("hidden");
  }

  function submitLearningStyleSurvey() {
    // Collect answers
    const answers = [];
    for (let i = 0; i < currentSurvey.questions.length; i++) {
      const selectedRadio = document.querySelector(
        `input[name="question-${i}"]:checked`
      );
      if (!selectedRadio) {
        alert("Please answer all questions before submitting.");
        return;
      }
      answers.push(selectedRadio.value);
    }

    // Submit answers to server
    fetch("/api/learning-style-survey", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        survey: currentSurvey,
        answers: answers,
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        // Display result
        learningStyleResult.textContent = `You are primarily a ${data.learning_style} learner.`;
        surveyResult.classList.remove("hidden");

        // Update profile display
        profileLearningStyle.textContent = data.learning_style;

        // Update next step button and current step
        nextStepBtn.textContent = "Continue to Document Upload";
        currentStep = "learning-style";
        updateProgressionUI();

        // Activate the upload section for the next step
        setTimeout(() => {
          // Show the next step button
          nextStepBtn.style.display = "block";
        }, 2000);
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  }

  function fetchPretest() {
    fetch("/api/pretest")
      .then((response) => response.json())
      .then((data) => {
        displayPretest(data);
        currentStep = "pretest";
        updateProgressionUI();
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  }

  function displayPretest(test) {
    // Update header
    pretestTitle.textContent = test.title;
    pretestDescription.textContent = test.description;

    // Clear previous questions
    pretestQuestions.innerHTML = "";

    // Create questions
    test.questions.forEach((question, index) => {
      const questionDiv = document.createElement("div");
      questionDiv.className = "question";

      const questionText = document.createElement("p");
      questionText.className = "question-text";
      questionText.textContent = `${index + 1}. ${question.question}`;
      questionDiv.appendChild(questionText);

      const choicesDiv = document.createElement("div");
      choicesDiv.className = "choices";

      question.choices.forEach((choice) => {
        const label = document.createElement("label");
        label.className = "choice-label";

        const radio = document.createElement("input");
        radio.type = "radio";
        radio.name = `pretest-question-${index}`;
        radio.value = choice[0]; // A, B, C, D

        label.appendChild(radio);
        label.appendChild(document.createTextNode(` ${choice}`));
        choicesDiv.appendChild(label);
      });

      questionDiv.appendChild(choicesDiv);
      pretestQuestions.appendChild(questionDiv);
    });

    // Hide results
    pretestResult.classList.add("hidden");

    // Show the section
    hideAllSections();
    pretestSection.classList.remove("hidden");
  }

  function submitPretest() {
    // Collect answers
    const answers = [];
    const numQuestions = pretestQuestions.querySelectorAll(".question").length;

    for (let i = 0; i < numQuestions; i++) {
      const selectedRadio = document.querySelector(
        `input[name="pretest-question-${i}"]:checked`
      );
      if (!selectedRadio) {
        alert("Please answer all questions before submitting.");
        return;
      }
      answers.push(selectedRadio.value);
    }

    // Submit answers to server
    fetch("/api/evaluate-pretest", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ answers: answers }),
    })
      .then((response) => response.json())
      .then((data) => {
        // Display result
        pretestScore.textContent = `${data.score} out of ${
          data.total
        } (${data.percentage.toFixed(1)}%)`;
        knowledgeLevel.textContent = data.knowledge_level;
        pretestResult.classList.remove("hidden");

        // Update profile display
        profileKnowledgeLevel.textContent = data.knowledge_level;

        // Update current step
        currentStep = "pretest";
        updateProgressionUI();

        // Scroll to the result
        pretestResult.scrollIntoView({ behavior: "smooth" });
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  }

  function fetchLearningPath() {
    fetch("/api/learning-path")
      .then((response) => response.json())
      .then((data) => {
        displayLearningPath(data);
        currentLearningPath = data;
        currentStep = "learning-path";
        updateProgressionUI();
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  }

  function displayLearningPath(learningPath) {
    // Update header
    learningPathTitle.textContent = learningPath.title;
    learningPathDescription.textContent = learningPath.description;

    // Clear previous content
    learningObjectivesList.innerHTML = "";
    modulesList.innerHTML = "";

    // Add learning objectives
    learningPath.objectives.forEach((objective) => {
      const li = document.createElement("li");
      li.textContent = objective;
      learningObjectivesList.appendChild(li);
    });

    // Add modules
    learningPath.modules.forEach((module, index) => {
      const moduleCard = document.createElement("div");
      moduleCard.className = "module-card";

      const moduleHeader = document.createElement("div");
      moduleHeader.className = "module-header";
      moduleHeader.innerHTML = `
        <h4>${module.title}</h4>
        <p>${module.description}</p>
      `;

      const moduleButton = document.createElement("button");
      moduleButton.className = "btn secondary-btn";
      moduleButton.textContent = "Start Module";
      moduleButton.addEventListener("click", function () {
        loadModule(index);
      });

      moduleCard.appendChild(moduleHeader);
      moduleCard.appendChild(moduleButton);
      modulesList.appendChild(moduleCard);
    });

    // Show the section
    hideAllSections();
    learningPathSection.classList.remove("hidden");
  }

  function loadModule(moduleIndex) {
    if (!currentLearningPath || !currentLearningPath.modules[moduleIndex]) {
      console.error("Module not found");
      return;
    }

    currentModuleIndex = moduleIndex;
    const module = currentLearningPath.modules[moduleIndex];

    // Extract topic from module title
    currentTopic = module.title.includes(": ")
      ? module.title.split(": ")[1]
      : module.title;

    // Fetch module content
    fetch(`/api/module-content/${moduleIndex}`)
      .then((response) => response.json())
      .then((data) => {
        // Update module display
        moduleTitle.textContent = module.title;
        moduleDescription.textContent = module.description;
        moduleContent.innerHTML = marked.parse(data.content);

        // Show the section
        hideAllSections();
        moduleContentSection.classList.remove("hidden");

        // Update current step
        currentStep = "modules";
        updateProgressionUI();
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  }

  function startDiscussion() {
    // Clear previous messages except for the welcome message
    discussionMessages.innerHTML = `
      <div class="message bot-message">
        Hi! I'm your learning partner. What would you like to discuss about ${currentTopic}?
      </div>
    `;

    // Reset input
    discussionInput.value = "";
    discussionInput.focus();

    // Show the discussion section
    hideAllSections();
    discussionSection.classList.remove("hidden");
  }

  function sendDiscussionMessage() {
    const message = discussionInput.value.trim();
    if (!message) return;

    // Add user message to chat
    addDiscussionMessage(message, "user");
    discussionInput.value = "";

    // Show loader
    discussionLoader.style.display = "block";
    discussionInput.disabled = true;
    sendDiscussionBtn.disabled = true;

    // Send message to server
    fetch("/api/peer-discussion", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        topic: currentTopic,
        message: message,
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        // Hide loader
        discussionLoader.style.display = "none";
        discussionInput.disabled = false;
        sendDiscussionBtn.disabled = false;

        // Add response to chat
        addDiscussionMessage(data.response, "bot");

        // Focus input for next message
        discussionInput.focus();
      })
      .catch((error) => {
        console.error("Error:", error);

        // Hide loader
        discussionLoader.style.display = "none";
        discussionInput.disabled = false;
        sendDiscussionBtn.disabled = false;

        // Add error message
        addDiscussionMessage(
          "Sorry, I encountered an error. Please try again.",
          "bot"
        );
      });
  }

  function addDiscussionMessage(text, sender) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${sender}-message`;
    messageDiv.innerHTML = marked.parse(text);
    discussionMessages.appendChild(messageDiv);
    discussionMessages.scrollTop = discussionMessages.scrollHeight;
  }

  function endDiscussion() {
    // Return to module content
    hideAllSections();
    moduleContentSection.classList.remove("hidden");
  }

  function takePosttest() {
    // Fetch posttest for current module
    fetch(`/api/posttest/${currentModuleIndex}`)
      .then((response) => response.json())
      .then((data) => {
        displayPosttest(data);
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  }

  function displayPosttest(test) {
    // Update header
    posttestTitle.textContent = test.title;
    posttestDescription.textContent = test.description;

    // Clear previous questions
    posttestQuestions.innerHTML = "";

    // Create questions
    test.questions.forEach((question, index) => {
      const questionDiv = document.createElement("div");
      questionDiv.className = "question";

      const questionText = document.createElement("p");
      questionText.className = "question-text";
      questionText.textContent = `${index + 1}. ${question.question}`;
      questionDiv.appendChild(questionText);

      const choicesDiv = document.createElement("div");
      choicesDiv.className = "choices";

      question.choices.forEach((choice) => {
        const label = document.createElement("label");
        label.className = "choice-label";

        const radio = document.createElement("input");
        radio.type = "radio";
        radio.name = `posttest-question-${index}`;
        radio.value = choice[0]; // A, B, C, D

        label.appendChild(radio);
        label.appendChild(document.createTextNode(` ${choice}`));
        choicesDiv.appendChild(label);
      });

      questionDiv.appendChild(choicesDiv);
      posttestQuestions.appendChild(questionDiv);
    });

    // Hide results
    posttestResult.classList.add("hidden");

    // Show the section
    hideAllSections();
    posttestSection.classList.remove("hidden");
  }

  function submitPosttest() {
    // Collect answers
    const answers = [];
    const numQuestions = posttestQuestions.querySelectorAll(".question").length;

    for (let i = 0; i < numQuestions; i++) {
      const selectedRadio = document.querySelector(
        `input[name="posttest-question-${i}"]:checked`
      );
      if (!selectedRadio) {
        alert("Please answer all questions before submitting.");
        return;
      }
      answers.push(selectedRadio.value);
    }

    // Submit answers to server
    fetch(`/api/evaluate-posttest/${currentModuleIndex}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ answers: answers }),
    })
      .then((response) => response.json())
      .then((data) => {
        // Display result
        posttestScore.textContent = `${data.score} out of ${
          data.total
        } (${data.percentage.toFixed(1)}%)`;
        newKnowledgeLevel.textContent = data.new_level;

        // Update level change message
        if (data.previous_level !== data.new_level) {
          levelChangeMessage.textContent = `Your knowledge level has changed from ${data.previous_level} to ${data.new_level}!`;
        } else {
          levelChangeMessage.textContent = `Your knowledge level remains ${data.new_level}.`;
        }

        posttestResult.classList.remove("hidden");

        // Update profile display
        profileKnowledgeLevel.textContent = data.new_level;

        // Scroll to the result
        posttestResult.scrollIntoView({ behavior: "smooth" });

        // Show learning log section after a delay
        setTimeout(() => {
          hideAllSections();
          learningLogSection.classList.remove("hidden");
        }, 5000);
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  }

  function submitLearningLog() {
    const content = learningLogContent.value.trim();

    if (!content) {
      alert("Please write your learning log reflection before submitting.");
      return;
    }

    fetch(`/api/learning-log/${currentModuleIndex}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ content: content }),
    })
      .then((response) => response.json())
      .then((data) => {
        // Display analysis
        understandingLevel.textContent = data.analysis.understanding_level;

        // Clear previous lists
        logStrengths.innerHTML = "";
        logAreas.innerHTML = "";
        logNextSteps.innerHTML = "";

        // Populate strengths
        data.analysis.strengths.forEach((strength) => {
          const li = document.createElement("li");
          li.textContent = strength;
          logStrengths.appendChild(li);
        });

        // Populate areas for improvement
        data.analysis.areas_for_improvement.forEach((area) => {
          const li = document.createElement("li");
          li.textContent = area;
          logAreas.appendChild(li);
        });

        // Populate next steps
        data.analysis.recommended_next_steps.forEach((step) => {
          const li = document.createElement("li");
          li.textContent = step;
          logNextSteps.appendChild(li);
        });

        // Show analysis
        logAnalysis.classList.remove("hidden");

        // Scroll to analysis
        logAnalysis.scrollIntoView({ behavior: "smooth" });

        // Update the profile display (this will fetch the latest profile)
        fetch("/api/profile")
          .then((response) => response.json())
          .then((profile) => {
            updateProfileDisplay(profile);
          })
          .catch((error) => {
            console.error("Error updating profile:", error);
          });
      })
      .catch((error) => {
        console.error("Error:", error);
        alert("Error submitting learning log. Please try again.");
      });
  }

  function askQuestion() {
    const question = questionInput.value.trim();
    if (question === "") return;

    addMessage(question, "user");
    questionInput.value = "";

    chatLoader.style.display = "block";
    questionInput.disabled = true;
    askBtn.disabled = true;

    fetch("/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question: question }),
    })
      .then((response) => response.json())
      .then((data) => {
        chatLoader.style.display = "none";
        questionInput.disabled = false;
        askBtn.disabled = false;

        if (data.answer) {
          addMessage(data.answer, "bot");
        } else {
          addMessage(
            "I encountered an error: " + (data.error || "Unknown error"),
            "bot"
          );
        }

        questionInput.focus();
      })
      .catch((error) => {
        chatLoader.style.display = "none";
        questionInput.disabled = false;
        askBtn.disabled = false;
        addMessage("Error: " + error.message, "bot");
      });
  }

  function addMessage(text, sender) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${sender}-message`;
    messageDiv.innerHTML = marked.parse(text);
    messages.appendChild(messageDiv);
    messages.scrollTop = messages.scrollHeight;
  }

  // Add event listeners for posttest and learning log
  takePosttestBtn.addEventListener("click", function () {
    takePosttest();
  });
});
