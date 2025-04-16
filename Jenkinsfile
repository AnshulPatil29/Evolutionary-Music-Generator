// --- START OF FILE Jenkinsfile.txt --- // Final Corrected Version //

pipeline {
    agent any
    triggers {
        pollSCM('H/5 * * * *') // Poll the repository every 5 minutes
    }
    environment {
        IMAGE_NAME = "evolutionary-music-generator"
        CONTAINER_NAME = "music-generator-container" // The fixed name for the container
    }
    stages {
        stage('Checkout') {
            steps {
                // Implicit checkout usually happens, this might be redundant.
                git url: 'https://github.com/AnshulPatil29/Evolutionary-Music-Generator.git', branch: 'main'
            }
        }
        stage('Build Docker Image') {
            steps {
                script {
                    echo "Building Docker image: ${IMAGE_NAME}:${env.BUILD_NUMBER}"
                    // Using Docker Pipeline plugin syntax (preferred if available)
                    docker.build("${IMAGE_NAME}:${env.BUILD_NUMBER}")
                    // OR using shell/bat command:
                    // bat "docker build -t ${IMAGE_NAME}:${env.BUILD_NUMBER} ."
                }
            }
        }
        // --- CORRECTED STAGE ---
        stage('Clean Up Existing Container') {
            steps {
                script {
                    echo "Attempting to remove any existing container named '${CONTAINER_NAME}'..."
                    // Use Windows Batch %VARIABLE% syntax inside bat step
                    bat '''
                        @echo off
                        docker rm -f %CONTAINER_NAME%
                        echo Container removal attempt finished.
                        exit /B 0
                    '''
                }
            }
        }
        // --- END OF CORRECTION ---
        stage('Run Docker Container') {
            steps {
                script {
                    echo "Running new container '${CONTAINER_NAME}' from image ${IMAGE_NAME}:${env.BUILD_NUMBER}"
                    // Use Windows Batch %VARIABLE% syntax here too for consistency
                    bat "docker run -d --name %CONTAINER_NAME% -p 8501:8501 ${IMAGE_NAME}:${env.BUILD_NUMBER}"
                    // Note: Groovy interpolation "${VAR}" *might* also work here directly in the bat string argument,
                    // but using %VAR% is safer within the bat context.
                }
            }
        }
    }
    post {
        always {
            echo 'Pipeline execution completed.'
        }
        success {
            echo "Pipeline finished successfully for build ${env.BUILD_NUMBER}."
        }
        failure {
            echo "Pipeline failed on build ${env.BUILD_NUMBER}."
        }
    }
}
// --- END OF FILE Jenkinsfile.txt ---