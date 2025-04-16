// --- START OF FILE Jenkinsfile.txt --- // Webhook Trigger Version //

pipeline {
    agent any
    // triggers { // REMOVE OR COMMENT OUT THIS BLOCK
    //     pollSCM('H/5 * * * *')
    // }
    environment {
        IMAGE_NAME = "evolutionary-music-generator"
        CONTAINER_NAME = "music-generator-container"
    }
    stages {
        stage('Checkout') {
            steps {
                // Ensure this checkout step uses the correct branch ('main')
                // The checkout mechanism triggered by the webhook will handle getting the correct commit.
                git url: 'https://github.com/AnshulPatil29/Evolutionary-Music-Generator.git', branch: 'main'
            }
        }
        stage('Build Docker Image') {
            steps {
                script {
                    echo "Building Docker image: ${IMAGE_NAME}:${env.BUILD_NUMBER}"
                    docker.build("${IMAGE_NAME}:${env.BUILD_NUMBER}")
                }
            }
        }
        stage('Clean Up Existing Container') {
            steps {
                script {
                    echo "Attempting to remove any existing container named '${CONTAINER_NAME}'..."
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
            steps {
                script {
                    echo "Running new container '${CONTAINER_NAME}' from image ${IMAGE_NAME}:${env.BUILD_NUMBER}"
                    bat "docker run -d --name %CONTAINER_NAME% -p 8501:8501 ${IMAGE_NAME}:${env.BUILD_NUMBER}"
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