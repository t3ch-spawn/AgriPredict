import React from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export default function SelectBox({
  theme,
  className,
  valueChange,
  selectList,
  defaultValue,
}) {
  return (
    <Select
      onValueChange={(value) => {
        valueChange(value);
      }}
      defaultValue={defaultValue}
    >
      <SelectTrigger
        className={`w-[180px] !border-greenText !text-greenText hover:shadow-xl duration-300 ${className}`}
      >
        <SelectValue placeholder={theme} />
      </SelectTrigger>
      <SelectContent className="bg-greenSide !border-greenText">
        {selectList.map((item) => {
          return (
            <SelectItem
              className="duration-300 cursor-pointer"
              value={item.value}
            >
              {item.name}
            </SelectItem>
          );
        })}
      </SelectContent>
    </Select>
  );
}
