// --- START OF FILE Jenkinsfile.txt --- // Fixed Version //

pipeline {
    agent any
    triggers {
        pollSCM('H/5 * * * *') // Poll the repository every 5 minutes
    }
    environment {
        IMAGE_NAME = "evolutionary-music-generator"
        CONTAINER_NAME = "music-generator-container"
    }
    stages {
        stage('Checkout') {
            steps {
                // Use the default checkout provided by the pipeline implicitly,
                // or keep this explicit one if needed for specific reasons.
                // Note: The log shows checkout happens twice, once implicitly and once here.
                // Consider removing this stage if the implicit checkout is sufficient.
                git url: 'https://github.com/AnshulPatil29/Evolutionary-Music-Generator.git', branch: 'main'
            }
        }
        stage('Build Docker Image') {
            steps {
                script {
                    // Use the Docker Pipeline plugin syntax for building
                    docker.build("${IMAGE_NAME}:${env.BUILD_NUMBER}")
                    // Alternatively, use bat if Docker Pipeline plugin is not configured/preferred:
                    // bat "docker build -t ${IMAGE_NAME}:${env.BUILD_NUMBER} ."
                }
            }
        }
        // --- CORRECTED STAGE ---
        stage('Clean Up Existing Container') { // Renamed stage for clarity
            steps {
                script {
                    // Force remove the container by name, ignoring errors if it doesn't exist.
                    // The 'exit /B 0' ensures this step succeeds regardless.
                    bat '''
                        @echo off
                        echo Attempting to remove container named ${CONTAINER_NAME}...
                        docker rm -f ${CONTAINER_NAME}
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
                    // Run the new container, mapping container port 8501 to host port 8501.
                    bat "docker run -d --name ${CONTAINER_NAME} -p 8501:8501 ${IMAGE_NAME}:${env.BUILD_NUMBER}"
                }
            }
        }
    }
    post {
        always {
            echo 'Pipeline execution completed.'
            // Optional: Add cleanup for old images if desired
            // script {
            //     bat "docker image prune -f"
            // }
        }
        // Add failure/success steps if needed
        // success {
        //     echo 'Pipeline finished successfully.'
        // }
        // failure {
        //     echo 'Pipeline failed.'
        // }
    }
}
// --- END OF FILE Jenkinsfile.txt ---