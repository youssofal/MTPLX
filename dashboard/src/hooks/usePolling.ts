import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";

export function useRecentRequests() {
  return useQuery({
    queryKey: ["metrics"],
    queryFn: api.getMetrics,
    refetchInterval: 1000,
    refetchOnWindowFocus: false,
  });
}

export function useSessionList() {
  return useQuery({
    queryKey: ["sessions"],
    queryFn: api.getSessions,
    refetchInterval: 2000,
    refetchOnWindowFocus: false,
  });
}

export function usePrefillHistory() {
  return useQuery({
    queryKey: ["prefillHistory"],
    queryFn: api.getPrefillHistory,
    refetchInterval: 5000,
    refetchOnWindowFocus: false,
  });
}

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: api.getHealth,
    refetchInterval: 10000,
    refetchOnWindowFocus: false,
  });
}

export function useRuntimeSystems() {
  return useQuery({
    queryKey: ["runtimeSystems"],
    queryFn: api.getSystems,
    refetchInterval: 3000,
    refetchOnWindowFocus: false,
  });
}
