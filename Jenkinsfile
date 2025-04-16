// --- START OF FILE Jenkinsfile.txt --- // Final Version with Testing //

pipeline {
    agent any // Runs on any available agent
    // No 'triggers' block if using GitHub webhooks (configured in Jenkins UI & GitHub)
    environment {
        IMAGE_NAME = "evolutionary-music-generator"
        CONTAINER_NAME = "music-generator-container" // Fixed name for the running container
        TEST_REPORT_FILE = 'test-results.xml'         // Name for the JUnit test report
    }
    stages {
        stage('Checkout') {
            // Checks out the source code from the specified branch
            steps {
                git url: 'https://github.com/AnshulPatil29/Evolutionary-Music-Generator.git', branch: 'main'
            }
        }
        stage('Build Docker Image') {
            // Builds the Docker image using the Dockerfile in the workspace.
            // Assumes 'pytest' is added to requirements.txt so it's included in the image.
            steps {
                script {
                    echo "Building Docker image: ${IMAGE_NAME}:${env.BUILD_NUMBER}"
                    // Use Docker Pipeline plugin integration (preferred)
                    docker.build("${IMAGE_NAME}:${env.BUILD_NUMBER}")
                    // OR, use shell/bat command:
                    // bat "docker build -t ${IMAGE_NAME}:${env.BUILD_NUMBER} ."
                }
            }
        }
        stage('Test') {
            // Runs automated tests using pytest inside the built Docker container
            steps {
                script {
                    echo "Running tests using image ${IMAGE_NAME}:${env.BUILD_NUMBER}"
                    // Run pytest inside the container. Mount workspace to /app (adjust if WORKDIR differs).
                    // Explicitly tell pytest where the tests are (/app/tests).
                    // Generate JUnit XML report inside the container at /app/test-results.xml,
                    // which maps back to the workspace due to the volume mount.
                    // Use bat for Windows agent.
                    bat """
                        docker run --rm ^
                          -v "%WORKSPACE%:/app" ^
                          ${IMAGE_NAME}:${env.BUILD_NUMBER} ^
                          pytest /app/tests --junitxml=/app/${TEST_REPORT_FILE}
                    """
                    // Note: Caret (^) is for line continuation in Windows Batch.
                    // If pytest fails, it returns non-zero, failing this stage.
                }
            }
            post {
                // Always archive test results using the JUnit plugin, regardless of test pass/fail.
                always {
                    echo "Archiving test results from ${TEST_REPORT_FILE}..."
                    junit "${TEST_REPORT_FILE}"
                }
            }
        }
        stage('Clean Up Existing Container') {
            // Forcefully removes any container with the target name BEFORE running a new one.
            // This prevents the "container name already in use" error.
            steps {
                script {
                    echo "Attempting to remove any existing container named '${CONTAINER_NAME}'..."
                    // Use Windows Batch %VARIABLE% syntax inside bat step.
                    // exit /B 0 ensures this step succeeds even if the container didn't exist.
                    bat '''
                        @echo off
                        docker rm -f %CONTAINER_NAME%
                        echo Container removal attempt finished.
                        exit /B 0
                    '''
                }
            }
        }
        stage('Run Docker Container') {
            // Runs the newly built and tested Docker image as a detached container.
            // This stage only runs if the 'Test' stage succeeded.
            steps {
                script {
                    echo "Running new container '${CONTAINER_NAME}' from image ${IMAGE_NAME}:${env.BUILD_NUMBER}"
                    // Use Windows Batch %VARIABLE% syntax.
                    bat "docker run -d --name %CONTAINER_NAME% -p 8501:8501 ${IMAGE_NAME}:${env.BUILD_NUMBER}"
                }
            }
        }
    }
    post {
        // Actions to run at the end of the pipeline run
        always {
            echo 'Pipeline execution finished.'
            // Optional: Add cleanup like pruning old Docker images
            // script {
            //    echo "Pruning unused Docker images..."
            //    bat "docker image prune -af" // -a to remove unused, -f to force
            // }
        }
        success {
            echo "Pipeline finished successfully for build ${env.BUILD_NUMBER}."
        }
        failure {
            // This block executes if any stage fails
            echo "Pipeline failed on build ${env.BUILD_NUMBER}."
            // Consider adding notifications (email, Slack) on failure here
        }
    }
}
// --- END OF FILE Jenkinsfile.txt ---