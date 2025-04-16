// --- START OF FILE Jenkinsfile.txt --- // Final Version //

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
                // This explicit checkout might be redundant if the default SCM checkout is sufficient.
                // Check your job configuration. If Jenkins checks out automatically, you might remove this stage.
                git url: 'https://github.com/AnshulPatil29/Evolutionary-Music-Generator.git', branch: 'main'
            }
        }
        stage('Build Docker Image') {
            // Builds a new image tagged with the unique build number
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
        stage('Clean Up Existing Container') {
            // Ensures any container with the target name is removed BEFORE trying to create a new one
            steps {
                script {
                    echo "Attempting to remove any existing container named '${CONTAINER_NAME}'..."
                    // Force remove the container by name.
                    // exit /B 0 ensures this step passes even if the container didn't exist.
                    bat '''
                        @echo off
                        docker rm -f ${CONTAINER_NAME}
                        echo Container removal attempt finished.
                        exit /B 0
                    '''
                }
            }
        }
        stage('Run Docker Container') {
            // Runs a new container using the newly built image and the cleaned-up container name
            steps {
                script {
                    echo "Running new container '${CONTAINER_NAME}' from image ${IMAGE_NAME}:${env.BUILD_NUMBER}"
                    bat "docker run -d --name ${CONTAINER_NAME} -p 8501:8501 ${IMAGE_NAME}:${env.BUILD_NUMBER}"
                }
            }
        }
    }
    post {
        always {
            echo 'Pipeline execution completed.'
            // Optional: Clean up dangling Docker images to save disk space
            // script {
            //    echo "Pruning unused Docker images..."
            //    bat "docker image prune -f"
            // }
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