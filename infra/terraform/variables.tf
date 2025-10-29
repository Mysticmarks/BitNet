variable "kubeconfig" {
  description = "Path to the kubeconfig file used for authentication"
  type        = string
  default     = "~/.kube/config"
}

variable "kube_context" {
  description = "Optional kubeconfig context to target"
  type        = string
  default     = null
}

variable "namespace" {
  description = "Namespace where the BitNet runtime will be deployed"
  type        = string
  default     = "bitnet"
}

variable "image" {
  description = "Container image to deploy"
  type        = string
  default     = "ghcr.io/microsoft/bitnet-runtime:latest"
}

variable "image_pull_policy" {
  description = "Image pull policy for the BitNet runtime"
  type        = string
  default     = "IfNotPresent"
}

variable "replicas" {
  description = "Number of BitNet runtime replicas"
  type        = number
  default     = 1
}

variable "model_url" {
  description = "Location of the GGUF model artefact"
  type        = string
}

variable "inference_args" {
  description = "Additional arguments passed to run_inference_server.py"
  type        = string
  default     = "--threads auto --diagnostics"
}

variable "container_port" {
  description = "Container port exposed by the runtime"
  type        = number
  default     = 8080
}

variable "service_port" {
  description = "Cluster service port"
  type        = number
  default     = 80
}

variable "service_type" {
  description = "Kubernetes service type"
  type        = string
  default     = "ClusterIP"
}

variable "resource_requests" {
  description = "Requested CPU/Memory resources"
  type        = map(string)
  default = {
    cpu    = "500m"
    memory = "2Gi"
  }
}

variable "resource_limits" {
  description = "CPU/Memory limits"
  type        = map(string)
  default = {
    cpu    = "2000m"
    memory = "8Gi"
  }
}
