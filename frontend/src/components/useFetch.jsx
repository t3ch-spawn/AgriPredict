import { useState, useEffect } from "react";

export const API_URL = "127.0.0.1";

export function useFetch(url, shouldRun = true) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = async () => {
    try {
      const res = await fetch(`http://${API_URL}:8000${url}`);
      if (!res.ok) throw new Error(`HTTP error! Status: ${res.status}`);
      const result = await res.json();
      setData(result);
      return result;
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  useEffect(() => {
    if (!shouldRun) return;
    fetchData();
  }, [url, shouldRun]);

  return { data, loading, error, refetch: fetchData };
}
