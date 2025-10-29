output "namespace" {
  description = "Namespace hosting the BitNet runtime"
  value       = kubernetes_namespace.bitnet.metadata[0].name
}

output "service_name" {
  description = "Name of the Kubernetes Service"
  value       = kubernetes_service.bitnet.metadata[0].name
}

output "service_port" {
  description = "Exposed service port"
  value       = kubernetes_service.bitnet.spec[0].port[0].port
}
