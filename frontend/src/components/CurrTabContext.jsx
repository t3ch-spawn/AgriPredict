/* eslint-disable */

import { createContext, useContext, useState } from "react";

const CurrTabContext = createContext();

export const CurrTabProvider = ({ children }) => {
  const [currTab, setCurrTab] = useState("overview");
  const [currState, setCurrState] = useState("Borno");
  const [currCommodity, setCurrCommodity] = useState("Rice (local)");

  return (
    <CurrTabContext.Provider
      value={{
        currTab,
        setCurrTab,
        currState,
        setCurrState,
        currCommodity,
        setCurrCommodity,
      }}
    >
      {children}
    </CurrTabContext.Provider>
  );
};

export function useCurrTab() {
  return useContext(CurrTabContext);
}
