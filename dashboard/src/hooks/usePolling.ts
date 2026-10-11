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

// Captured once at server startup; the slow refetch only notices a restart.
export function useLaunchInfo() {
  return useQuery({
    queryKey: ["launch"],
    queryFn: api.getLaunch,
    refetchInterval: 30000,
    refetchOnWindowFocus: false,
  });
}
